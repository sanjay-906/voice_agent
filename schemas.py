from pydantic import BaseModel, Field
from typing import Optional, List


class Medication(BaseModel):
    """Medication information"""
    name: str = Field(description="Medication name (brand or generic)")
    dose: Optional[str] = Field(None, description="Dosage (e.g., '500mg')")
    frequency: Optional[str] = Field(None, description="How often taken")
    route: Optional[str] = Field("oral", description="Administration route")
    indication: Optional[str] = Field(None, description="What it's for")
    adherence: Optional[str] = Field(None, description="Compliance status")
    effectiveness: Optional[str] = Field(None, description="How well it works")


class Allergy(BaseModel):
    """Allergy information"""
    allergen: str = Field(description="What causes allergy")
    reaction: List[str] = Field(description="Symptoms/reactions")
    severity: str = Field(description="mild, moderate, serious, or life-threatening")
    requires_emergency_treatment: bool = Field(False, description="Needs EpiPen/ER")


class ChiefComplaint(BaseModel):
    """Primary reason for visit"""
    complaint: str = Field(description="Main symptom/issue")
    duration: Optional[str] = Field(None, description="How long")
    severity: Optional[str] = Field(None, description="Severity rating")
    onset: Optional[str] = Field(None, description="sudden or gradual")
    location: Optional[str] = Field(None, description="Body location")


class PatientInfo(BaseModel):
    """Patient demographics"""
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class PresentIllness(BaseModel):
    """Current illness details"""
    chief_complaints: List[ChiefComplaint] = []
    symptoms: List[str] = []
    timeline: Optional[str] = None


class PastMedicalHistory(BaseModel):
    """Past medical conditions"""
    conditions: List[str] = []
    surgeries: List[str] = []
    hospitalizations: List[str] = []


class FamilyHistory(BaseModel):
    """Family medical history"""
    conditions: List[str] = []


class SocialHistory(BaseModel):
    """Social and lifestyle factors"""
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    drugs: Optional[str] = None
    occupation: Optional[str] = None
    exercise: Optional[str] = None


class MedicalIntake(BaseModel):
    """Complete medical intake data"""
    patient_info: Optional[PatientInfo] = None
    present_illness: Optional[PresentIllness] = None
    medications: List[Medication] = []
    allergies: List[Allergy] = []
    past_medical_history: Optional[PastMedicalHistory] = None
    family_history: Optional[FamilyHistory] = None
    social_history: Optional[SocialHistory] = None
