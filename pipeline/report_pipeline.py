from pathlib import Path
from core.schemas import EvidenceResult
from processing.document_processor import DocumentProcessor
from pipeline.clinical_pipeline import ClinicalPipeline

class ReportPipeline:
    def __init__(self,document_processor=None,clinical_pipeline=None,):
        self.document_processor = (
            document_processor or DocumentProcessor()
        )

        self.clinical_pipeline = (
            clinical_pipeline or ClinicalPipeline()
        )

    def process(self,file_path: str | Path,top_k:int=3)->dict:
        report_text = self.document_processor.extract_text(file_path)
        clinical_result, evidence_result = (self.clinical_pipeline.process(report_text,top_k=top_k,))
        return {
            "filename": Path(file_path).name,
            "extracted_text": report_text,
            "clinical_result": clinical_result,
            "evidence_result": evidence_result,
        }