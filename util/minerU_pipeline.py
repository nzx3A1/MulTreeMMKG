import os
import sys
import argparse
import time
import uuid
import json
import re
import requests
import zipfile
import io
from pathlib import Path
from pypdf import PdfReader, PdfWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_config import settings as model_settings

# ----------------- 配置信息 -----------------
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdf"
MINERU_OUTPUT_DIR = PROJECT_ROOT / "data" / "mineru_output"
API_BATCH_URL = model_settings.mineru.batch_url
API_EXTRACT_URL_TEMPLATE = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"
MAX_PAGES_PER_PDF = 200
MAX_FILES_PER_BATCH = 20
SPLIT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "mineru_splits")
HEADER = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {model_settings.mineru.token}"
}


def to_windows_long_path(path):
    """在 Windows 下将绝对路径转换为 long path 形式，避免超长路径打开失败。"""
    if os.name != "nt":
        return path

    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path[2:]
    return "\\\\?\\" + abs_path


def can_open_file(path):
    """检查文件是否存在且可读。"""
    try:
        with open(to_windows_long_path(path), "rb"):
            return True
    except OSError:
        return False


def ensure_directory(path):
    """创建目录，Windows 下自动切换到 long path 形式。"""
    os.makedirs(to_windows_long_path(path), exist_ok=True)


def get_pdf_page_count(path):
    """获取 PDF 页数，失败时返回 None。"""
    try:
        with open(to_windows_long_path(path), "rb") as pdf_file:
            return len(PdfReader(pdf_file).pages)
    except Exception as e:
        print(f"无法读取页数: {path} -> {e}")
        return None


def split_pdf_if_needed(path, max_pages=MAX_PAGES_PER_PDF):
    """如果 PDF 页数过大则自动拆分为多个临时文件，返回可上传文件列表。"""
    page_count = get_pdf_page_count(path)
    if page_count is None or page_count <= max_pages:
        return [{
            "path": path,
            "upload_name": os.path.basename(path),
            "origin_dir": os.path.dirname(path),
            "source_path": path,
        }]

    ensure_directory(SPLIT_CACHE_DIR)

    base_name = os.path.splitext(os.path.basename(path))[0]
    source_dir = os.path.dirname(path)
    split_items = []

    print(f"文件页数过大，自动拆分: {os.path.basename(path)} | 页数 {page_count} | 每份最多 {max_pages} 页")

    try:
        with open(to_windows_long_path(path), "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            part_index = 1
            for start_index in range(0, page_count, max_pages):
                writer = PdfWriter()
                end_index = min(start_index + max_pages, page_count)

                for page_index in range(start_index, end_index):
                    writer.add_page(reader.pages[page_index])

                split_file_name = f"{base_name}_part{part_index:03d}.pdf"
                split_path = os.path.join(SPLIT_CACHE_DIR, split_file_name)
                with open(to_windows_long_path(split_path), "wb") as output_file:
                    writer.write(output_file)

                split_items.append({
                    "path": split_path,
                    "upload_name": split_file_name,
                    "origin_dir": source_dir,
                    "source_path": path,
                })
                part_index += 1
    except Exception as e:
        print(f"拆分失败，回退为原文件上传: {path} -> {e}")
        return [{
            "path": path,
            "upload_name": os.path.basename(path),
            "origin_dir": os.path.dirname(path),
            "source_path": path,
        }]

    print(f"拆分完成: {os.path.basename(path)} -> {len(split_items)} 个分片")
    return split_items


def expand_pdf_upload_items(pdf_paths):
    """展开待上传文件，必要时将大 PDF 自动拆分。"""
    upload_items = []
    for path in pdf_paths:
        upload_items.extend(split_pdf_if_needed(path))
    return upload_items


def chunk_upload_items(upload_items, chunk_size=MAX_FILES_PER_BATCH):
    """将待上传文件按单批最大数量切分。"""
    for start_index in range(0, len(upload_items), chunk_size):
        yield upload_items[start_index:start_index + chunk_size]


def find_pdf_files_recursive(root_dir):
    """递归查找目录下的所有 .pdf 文件，并返回完整路径列表"""
    pdf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def filter_pdffiles(file_paths, output_root):
    """过滤已经解析过的 PDF，并跳过 MinerU 中间文件命名格式，避免重复提交。"""
    #过滤pdf文件，判断当前文件所在文件夹下是否已经存在同名的文件夹，如果存在则认为该pdf文件已经处理过了，无需再次提交
    #同时也过滤掉类似97b6dde8-a3bb-4464-8c2c-ec8f849c68ba_origin.pdf，72beb496-99a3-4928-b39b-ea14747f3c47_origin.pdf'这种命名格式的文件，避免误提交

    filtered_files = []
    uuid_origin_pattern = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_origin\.pdf$")
    
    for path in file_paths:
        file_name = os.path.basename(path)

        if uuid_origin_pattern.match(file_name):
            print(f"文件 {file_name} 命名符合 *_origin 中间文件规则，跳过提交")
            continue

        folder_name = os.path.splitext(file_name)[0]
        folder_path = os.path.join(str(output_root), folder_name)
        if not has_mineru_result(folder_path):
            filtered_files.append(path)
        else:
            print(f"文件 {file_name} 已经处理过了，跳过提交")
    return filtered_files


def has_mineru_result(folder_path):
    """判断输出目录中是否已经存在 MinerU 核心产物，避免空目录导致误判为已完成。"""
    if not os.path.isdir(folder_path):
        return False
    required_candidates = ("full.md", "content_list.json", "layout.json")
    return any(os.path.exists(os.path.join(folder_path, name)) for name in required_candidates)


def clean_md_tags(md_path):
    """清理 md 文件中的 <sup>, </sup>, <sub>, </sub> 标签。"""
    try:
        with open(to_windows_long_path(md_path), "r", encoding="utf-8") as f:
            content = f.read()
        
        content = content.replace("<sup>", "").replace("</sup>", "").replace("<sub>", "").replace("</sub>", "")
        
        with open(to_windows_long_path(md_path), "w", encoding="utf-8") as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"  清理标签失败 {md_path}: {e}")
        return False


def batch_clean_full_md(output_root):
    """批量清理所有 full.md 文件中的标签。"""
    cleaned_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(output_root):
        for file in files:
            if file == "full.md":
                md_path = os.path.join(root, file)
                if os.path.isfile(md_path):
                    print(f"清理标签: {md_path}")
                    if clean_md_tags(md_path):
                        cleaned_count += 1
                    else:
                        skipped_count += 1
    
    print(f"\n批量清理完成: 成功 {cleaned_count} 个文件，失败/跳过 {skipped_count} 个文件")
    return cleaned_count


def download_with_retries(url, retries=3, timeout=120):
    """直连下载 MinerU 结果 zip，避免系统代理导致 CDN TLS 握手中断。"""
    last_error = None
    # MinerU 的签名下载地址可直接访问；禁用环境代理可规避代理链路上的 SSL EOF。
    session = requests.Session()
    session.trust_env = False
    for attempt in range(1, retries + 1):
        verify = attempt != retries
        try:
            return session.get(url, stream=True, timeout=timeout, verify=verify)
        except requests.RequestException as e:
            last_error = e
            print(f"  下载重试 {attempt}/{retries} 失败: {e}")
            time.sleep(min(5 * attempt, 15))
    raise last_error



def submit_files_and_upload(pdf_paths_or_items):
    """提交文件列表获取上传URL，并执行上传，返回 batch_id"""
    if not pdf_paths_or_items:
        print("未找到需要处理的PDF文件")
        return None

    if isinstance(pdf_paths_or_items[0], dict):
        upload_items = pdf_paths_or_items
    else:
        upload_items = expand_pdf_upload_items(pdf_paths_or_items)

    # 提交前再次校验文件可读，避免扫描后文件被移动/删除或路径过长导致中断
    valid_pdf_paths = []
    valid_upload_items = []
    skipped_missing = 0
    for item in upload_items:
        path = item["path"]
        if can_open_file(path):
            valid_pdf_paths.append(path)
            valid_upload_items.append(item)
        else:
            skipped_missing += 1
            print(f"跳过不可读文件: {path}")

    if skipped_missing:
        print(f"提交前校验: 跳过 {skipped_missing} 个不可读文件")

    if not valid_pdf_paths:
        print("没有可提交的PDF文件（全部不可读或不存在）")
        return None

    # 1. 构造请求参数
    files_list = []
    for item in valid_upload_items:
        file_name = item["upload_name"]
        data_id = str(uuid.uuid4())
        files_list.append({
            "name": file_name,
            "data_id": data_id
        })

    data = {
        "files": files_list,
        "model_version": "vlm",
        # 关闭 MinerU 表格识别，避免将表格内容解析为结构化表格。
        "enable_table": False,
    }

    print(f"正在提交 {len(valid_upload_items)} 个文件进行处理...")
    
    # 2. 调用批量处理接口
    try:
        response = requests.post(API_BATCH_URL, headers=HEADER, json=data)
        if response.status_code != 200:
            print(f"提交失败: {response.text}")
            return None
            
        result = response.json()
        if result["code"] != 0:
            print(f"提交返回错误: {result.get('msg')}")
            return None

        batch_data = result["data"]
        batch_id = batch_data["batch_id"]
        urls = batch_data["file_urls"]
        
        print(f"批次创建成功，batch_id: {batch_id}")
        
        # 3. 上传文件
        print("开始上传文件...")
        if len(urls) != len(valid_upload_items):
            print(f"警告: 返回上传URL数量({len(urls)})与文件数量({len(valid_upload_items)})不一致，将按较小数量处理")

        upload_count = min(len(urls), len(valid_upload_items))
        upload_failed = 0

        for i in range(upload_count):
            url = urls[i]
            file_item = valid_upload_items[i]
            file_path = file_item["path"]
            file_name = file_item["upload_name"]

            try:
                with open(to_windows_long_path(file_path), 'rb') as f:
                    res_upload = requests.put(url, data=f)
                    if res_upload.status_code == 200:
                        print(f"  [{i+1}/{upload_count}] 上传成功: {file_name}")
                    else:
                        upload_failed += 1
                        print(f"  [{i+1}/{upload_count}] 上传失败: {file_name} (Status: {res_upload.status_code})")
            except OSError as e:
                upload_failed += 1
                print(f"  [{i+1}/{upload_count}] 上传跳过: {file_name} (文件不可读: {e})")

        if upload_failed:
            print(f"上传阶段结束: 失败/跳过 {upload_failed} 个文件")
        
        return batch_id

    except Exception as e:
        print(f"提交上传过程中发生异常: {e}")
        return None

def download_completed_files(extract_results, output_root, file_origin_map=None):
    """下载并解压所有处理完成的文件"""
    print("\n开始下载处理结果...")
    
    for item in extract_results:
        state = item.get("state")
        file_name = item.get("file_name")
        full_zip_url = item.get("full_zip_url")
        
        if state == "done" and full_zip_url:
            try:
                # 构造解压目录名（去除 .pdf 后缀）
                folder_name = os.path.splitext(file_name)[0]
                
                # 确定保存目录：如果有原始映射则用原始目录，否则用output_root
                if file_origin_map and file_name in file_origin_map:
                    save_root = file_origin_map[file_name]
                else:
                    save_root = output_root

                target_dir = os.path.join(save_root, folder_name)
                
                # 如果目录不存在则创建
                    
                zip_file_name = folder_name + ".zip"
                zip_save_path = os.path.join(output_root, zip_file_name)
                
                print(f"正在下载: {file_name}")
                zip_resp = download_with_retries(full_zip_url)
                
                if zip_resp.status_code == 200:
                    # 保存 ZIP
                    with open(to_windows_long_path(zip_save_path), 'wb') as f:
                        for chunk in zip_resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # 解压
                    print(f"  正在解压到: {target_dir}")
                    ensure_directory(target_dir)
                    with zipfile.ZipFile(to_windows_long_path(zip_save_path), 'r') as z:
                        z.extractall(to_windows_long_path(target_dir))
                        
                    # 清理 full.md 中的标签
                    full_md_path = os.path.join(target_dir, "full.md")
                    if os.path.exists(full_md_path):
                        print(f"  清理标签: {full_md_path}")
                        clean_md_tags(full_md_path)
                        
                    # 可选：删除压缩包
                    # os.remove(zip_save_path)
                    print(f"  处理完成: {folder_name}")
                else:
                    print(f"  下载失败，状态码: {zip_resp.status_code}")
                    
            except Exception as e:
                print(f"  下载/解压异常 {file_name}: {e}")
        else:
            print(f"跳过文件 {file_name}，状态: {state}")

def query_batch_results(batch_id):
    """查询 MinerU 批次状态并返回 extract_result 列表，用于轮询和失败后的下载重试。"""
    url = API_EXTRACT_URL_TEMPLATE.format(batch_id=batch_id)
    res = requests.get(url, headers=HEADER, timeout=model_settings.mineru.timeout_secs)
    if res.status_code != 200:
        raise RuntimeError(f"查询状态失败: HTTP {res.status_code} {res.text}")

    result_json = res.json()
    if result_json.get("code") != 0:
        raise RuntimeError(f"API 返回错误: {result_json.get('msg')}")

    data = result_json.get("data", {})
    return data.get("extract_result", [])


def save_extract_results(batch_id, extract_results, output_root):
    """保存 MinerU 批次结果快照，方便 CDN 下载失败后直接按 batch_id 重试下载。"""
    ensure_directory(str(output_root))
    snapshot_path = Path(output_root) / f"extract_results_{batch_id}.json"
    with open(to_windows_long_path(str(snapshot_path)), "w", encoding="utf-8") as file:
        json.dump(extract_results, file, ensure_ascii=False, indent=2)
        file.write("\n")


def download_existing_batch(batch_id, output_root):
    """根据已有 batch_id 查询并下载结果，避免下载失败时重复上传和重复解析 PDF。"""
    extract_results = query_batch_results(batch_id)
    if not extract_results:
        print(f"批次 {batch_id} 暂无结果")
        return
    save_extract_results(batch_id, extract_results, output_root)
    download_completed_files(extract_results, str(output_root))


def check_status_loop(batch_id, output_dir, file_origin_map=None):
    """
    轮询检查批次状态：
    1. 每隔1分钟调用一次查询接口
    2. 判断所有文件是否都已完成 (done 或 failed)
    3. 如果全部完成，则下载结果并结束
    4. 否则继续等待
    """
    url = API_EXTRACT_URL_TEMPLATE.format(batch_id=batch_id)
    
    print("\n进入状态轮询阶段...")
    
    while True:
        # 等待 1 分钟
        print("等待 1 分钟后检查状态...")
        time.sleep(60)
        
        try:
            res = requests.get(url, headers=HEADER)
            if res.status_code != 200:
                print(f"查询状态失败: {res.status_code}，将在下一轮重试")
                continue
                
            result_json = res.json()
            if result_json.get("code") != 0:
                print(f"API 返回错误: {result_json.get('msg')}，将在下一轮重试")
                continue
            
            data = result_json.get("data", {})
            extract_results = data.get("extract_result", [])
            
            if not extract_results:
                print("未获取到结果列表，将在下一轮重试")
                continue

            # 检查状态
            all_finished = True
            processing_count = 0
            done_count = 0
            failed_count = 0
            
            for item in extract_results:
                state = item.get("state")
                if state == "done":
                    done_count += 1
                elif state == "failed":
                    failed_count += 1
                else:
                    # todo, processing, pending 等状态
                    all_finished = False
                    processing_count += 1
            
            print(f"当前进度: 总数 {len(extract_results)} | 完成 {done_count} | 失败 {failed_count} | 处理中 {processing_count}")
            
            if all_finished:
                print("所有文件处理结束，准备下载...")
                save_extract_results(batch_id, extract_results, output_dir)
                download_completed_files(extract_results, output_dir, file_origin_map)
                break
            else:
                print("存在未完成的文件，继续等待...")

        except Exception as e:
            print(f"轮询过程中发生异常: {e}")
            # 异常发生时继续循环，避免中断
            continue

def _legacy_main():
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录不存在 -> {DATA_DIR}")
        return

    # 1. 递归查找文件
    print(f"正在扫描目录: {DATA_DIR}")
    pdf_files = find_pdf_files_recursive(DATA_DIR)
    pdf_files = filter_pdffiles(pdf_files)
    
    if not pdf_files:
        print("未找到 PDF 文件")
        return
        
    print(f"本次将提交 {len(pdf_files)} 个 PDF 文件")
    
    # 2. 预处理并提交上传
    upload_items = expand_pdf_upload_items(pdf_files)
    if not upload_items:
        print("未生成可提交的文件列表")
        return

    batch_groups = list(chunk_upload_items(upload_items))
    print(f"将按每批最多 {MAX_FILES_PER_BATCH} 份文件分为 {len(batch_groups)} 轮提交")

    for batch_index, batch_items in enumerate(batch_groups, start=1):
        print(f"\n===== 第 {batch_index}/{len(batch_groups)} 轮提交 =====")
        batch_id = submit_files_and_upload(batch_items)

        if batch_id:
            # 构建文件名到原始目录的映射，拆分后的分片也回写到原始目录
            file_origin_map = {item["upload_name"]: item["origin_dir"] for item in batch_items}

            # 3. 轮询状态并下载
            check_status_loop(batch_id, DATA_DIR, file_origin_map)
        else:
            print(f"第 {batch_index} 轮提交失败，跳过后续轮询")

def write_output_index(pdf_files, output_root):
    """写出 MinerU 输出索引，记录原始 PDF 与预期解压目录，方便后续流水线定位解析结果。"""
    ensure_directory(str(output_root))
    rows = []
    for path in pdf_files:
        file_name = os.path.basename(path)
        paper_id = os.path.splitext(file_name)[0]
        rows.append({
            "paper_id": paper_id,
            "source_pdf": file_name,
            "source_path": str(path),
            "output_dir": str(Path(output_root) / paper_id),
        })

    index_path = Path(output_root) / "index.json"
    with open(to_windows_long_path(str(index_path)), "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    """主流程：扫描 data/raw_pdf 下的 PDF，提交 MinerU 解析，并把结果保存到 data/mineru_output。"""
    parser = argparse.ArgumentParser(description="Submit raw PDFs to MinerU and save outputs under data/mineru_output")
    parser.add_argument("--input", type=Path, default=RAW_PDF_DIR, help="PDF 输入目录")
    parser.add_argument("--output", type=Path, default=MINERU_OUTPUT_DIR, help="MinerU 结果输出目录")
    parser.add_argument("--batch-id", default=None, help="已有 MinerU batch_id；提供后只重试查询和下载结果")
    parser.add_argument("--clean-only", action="store_true", help="仅清理所有 full.md 文件中的标签，不提交新任务")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if args.clean_only:
        print(f"执行批量清理标签，输出目录: {output_dir}")
        batch_clean_full_md(str(output_dir))
        return

    if args.batch_id:
        ensure_directory(str(output_dir))
        download_existing_batch(args.batch_id, output_dir)
        return

    if not input_dir.exists():
        print(f"错误: PDF 输入目录不存在 -> {input_dir}")
        return

    ensure_directory(str(output_dir))

    # 递归查找 PDF 文件，并以 data/mineru_output 中的同名目录作为已处理判断依据。
    print(f"正在扫描 PDF 目录: {input_dir}")
    pdf_files = find_pdf_files_recursive(str(input_dir))
    pdf_files = filter_pdffiles(pdf_files, output_dir)

    if not pdf_files:
        print("未找到需要提交的 PDF 文件")
        return

    print(f"本次将提交 {len(pdf_files)} 个 PDF 文件，输出目录: {output_dir}")
    write_output_index(pdf_files, output_dir)

    # 预处理并提交上传；大 PDF 会先拆分为临时分片。
    upload_items = expand_pdf_upload_items(pdf_files)
    if not upload_items:
        print("未生成可提交的文件列表")
        return

    batch_groups = list(chunk_upload_items(upload_items))
    print(f"将按每批最多 {MAX_FILES_PER_BATCH} 份文件分为 {len(batch_groups)} 轮提交")

    for batch_index, batch_items in enumerate(batch_groups, start=1):
        print(f"\n===== 第 {batch_index}/{len(batch_groups)} 轮提交 =====")
        batch_id = submit_files_and_upload(batch_items)

        if batch_id:
            # 轮询状态并统一下载到 data/mineru_output，而不是写回 raw_pdf 目录。
            check_status_loop(batch_id, str(output_dir))
        else:
            print(f"第 {batch_index} 轮提交失败，跳过后续轮询")


if __name__ == "__main__":
    main()
