# Backbone adapters for YOLO-World backbone-swap experiments.
# Importing this package registers UltralyticsYOLOBackbone into the MMYOLO MODELS
# registry so it can be referenced from config files via custom_imports.
from .ultralytics_backbone import UltralyticsYOLOBackbone

__all__ = ["UltralyticsYOLOBackbone"]
