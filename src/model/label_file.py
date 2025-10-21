from pydantic import BaseModel


class LabelFileAnnotation(BaseModel):
    object_recognition: int
    text_language: int


class LabelFileDataset(BaseModel):
    category: int
    identifier: str
    label_path: str
    name: str
    src_path: str
    type: int


class LabelFileImages(BaseModel):
    acquistion_location: str
    application_field: str
    background: int
    data_captured: str
    height: int
    identifier: str
    media_type: int
    pen_color: str
    pen_type: int
    type: str
    width: int
    writer_age: int
    writer_sex: int
    written_content: int


class LabelFileBBoxItem(BaseModel):
    data: str
    id: int
    x: list[int]
    y: list[int]


class LabelFile(BaseModel):
    Annotation: LabelFileAnnotation
    Dataset: LabelFileDataset
    Images: LabelFileImages
    bbox: list[LabelFileBBoxItem]
