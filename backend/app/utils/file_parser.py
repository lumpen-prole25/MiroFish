"""
File Parsertool
supportPDF、Markdown、TXTfile's textextract
"""

import os
from pathlib import Path
from typing import List, Optional


def _read_text_with_fallback(file_path: str) -> str:
    """
    readtextfile, UTF-8failedtime自动探测编码。
    
    采用多级回退策略:
    1. firstattempt UTF-8 解码
    2. use charset_normalizer 检测编码
    3. 回退to  chardet 检测编码
    4. 最终use UTF-8 + errors='replace' 兜底
    
    Args:
        file_path: filepath
        
    Returns:
        解码后's textcontent
    """
    data = Path(file_path).read_bytes()
    
    # firstattempt UTF-8
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    # attemptuse charset_normalizer 检测编码
    encoding = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            encoding = best.encoding
    except Exception:
        pass
    
    # 回退to  chardet
    if not encoding:
        try:
            import chardet
            result = chardet.detect(data)
            encoding = result.get('encoding') if result else None
        except Exception:
            pass
    
    # 最终兜底:use UTF-8 + replace
    if not encoding:
        encoding = 'utf-8'
    
    return data.decode(encoding, errors='replace')


class FileParser:
    """File Parser器"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        fromfileextracttext
        
        Args:
            file_path: filepath
            
        Returns:
            extract's textcontent
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"filedoes not exist: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不support's fileformat: {suffix}")
        
        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)
        
        raise ValueError(f"none法process's fileformat: {suffix}")
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """fromPDFextracttext"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("需要安装PyMuPDF: pip install PyMuPDF")
        
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        """fromMarkdownextracttext, support自动编码检测"""
        return _read_text_with_fallback(file_path)
    
    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """fromTXTextracttext, support自动编码检测"""
        return _read_text_with_fallback(file_path)
    
    @classmethod
    def extract_from_multiple(cls, file_paths: List[str]) -> str:
        """
        from多 itemsfileextracttext并合并
        
        Args:
            file_paths: filepathlist
            
        Returns:
            合并后's text
        """
        all_texts = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                text = cls.extract_text(file_path)
                filename = Path(file_path).name
                all_texts.append(f"=== document {i}: {filename} ===\n{text}")
            except Exception as e:
                all_texts.append(f"=== document {i}: {file_path} (extractfailed: {str(e)}) ===")
        
        return "\n\n".join(all_texts)


def split_text_into_chunks(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[str]:
    """
    将text分割成小块
    
    Args:
        text: 原始text
        chunk_size: 每块's charactercount
        overlap: 重叠charactercount
        
    Returns:
        text块list
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # attemptin 句子edge界处分割
        if end < len(text):
            # 查找最近's 句子ending符
            for sep in ['。', '！', '？', '.\n', '!\n', '?\n', '\n\n', '. ', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 下one items块from重叠位置starting
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

