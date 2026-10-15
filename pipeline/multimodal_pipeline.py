from pathlib import Path
from core.config import DEFAULT_TOP_K
from core.exceptions import InputValidationError
from core.schemas import EvidenceResult, MultimodalContext
from pipeline.clinical_pipeline import ClinicalPipeline
from pipeline.image_pipeline import ImagePipeline
from pipeline.report_pipeline import ReportPipeline

class MultimodalPipeline:
    def __init__(self,clinical_pipeline=None,image_pipeline=None,report_pipeline=None,):
        self.clinical_pipeline = (
            clinical_pipeline or ClinicalPipeline()
        )

        self.image_pipeline = (
            image_pipeline or ImagePipeline()
        )

        self.report_pipeline = (
            report_pipeline
            or ReportPipeline(
                clinical_pipeline=self.clinical_pipeline
            )
        )