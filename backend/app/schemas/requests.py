from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator


class Registration(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)

    @field_validator("name")
    @classmethod
    def trim_name(cls, value):
        if not value.strip():
            raise ValueError("Name cannot be empty")
        return value.strip()


class Login(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    email: str


class Preferences(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    retain_images: bool = False

    @field_validator("name")
    @classmethod
    def trim_name(cls, value):
        return Registration.trim_name(value)


class Consent(BaseModel):
    essential: bool = True
    authentication: bool = True
    analytics: bool = False
