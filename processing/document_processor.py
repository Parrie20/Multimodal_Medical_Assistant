from pathlib import Path
from pypdf import PdfReader
from core.exceptions import (
    InputValidationError,
    UnsupportedFileTypeError,
)

class DocumentProcessor:
    EXTENSIONS=[".txt",".pdf"]

    def extracted_text(self,file_path:str | Path)->str:
        path=Path(file_path).expanduser()
        if path.suffix.lower() not in self.EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported document type: {path.suffix}"
            )
        if not path.is_file():
            raise InputValidationError(
                f"Document not found: {path}"
            )
        if path.stat().st_size==0:
            raise InputValidationError(
                f"The document is empty."
            )
        if path.suffix.lower()==".txt":
            text=path.read_text(encoding="utf-8")
        else:
            self.extract_pdf_text(path)
        text=text.strip()
        if not text:
            raise InputValidationError(
                "No text could be extracted from the document. "
                "It may be a scanned PDF requiring OCR."
            )
        return text

    def extract_pdf_text(self,path:Path)->str:
        reader=PdfReader(str(path))
        pages=[]
        for page in reader.pages:
            page_text=page.extract_text() or ""
            if page.text.strip():
                pages.append(page_text.strip())

        return "\n\n".join(pages)
