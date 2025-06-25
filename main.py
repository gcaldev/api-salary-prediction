import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union, Optional, Literal
from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder
from enum import Enum
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import category_encoders as ce
import pickle


app = FastAPI()

class SalaryPredictionResponseDTO(BaseModel):
    sugested_salary: int = Field(alias="sugested_salary", description="Suggested salary based on the model prediction", example=50000)

class ProfessionalProfile(BaseModel):
    experience_level: str
    employment_type: str
    job_title: str
    employee_residence: str
    remote_ratio: int
    company_location: str
    company_size: str


with open('encoder.pkl', 'rb') as f:
    binary_encoder = pickle.load(f)

job_mapping = {
    "Data Analyst": [
        "Data Analyst", "BI Analyst", "BI Data Analyst", "Business Data Analyst",
        "Marketing Data Analyst", "Finance Data Analyst", "Financial Data Analyst",
        "Insight Analyst", "Staff Data Analyst", "Product Data Analyst", "Principal Data Analyst",
        "Lead Data Analyst"
    ],
    "Data Scientist": [
        "Data Scientist", "Applied Data Scientist", "AI Scientist", "Research Scientist",
        "Data Science Consultant", "Data Science Engineer", "Data Science Lead",
        "Data Science Manager", "Data Science Tech Lead", "Data Scientist Lead",
        "Staff Data Scientist", "Product Data Scientist"
    ],
    "Data Engineer": [
        "Data Engineer", "Big Data Engineer", "Cloud Data Engineer", "Data Infrastructure Engineer",
        "Data DevOps Engineer", "Software Data Engineer", "ETL Developer", "ETL Engineer",
        "BI Developer", "Azure Data Engineer", "Marketing Data Engineer"
    ],
    "Machine Learning Engineer": [
        "ML Engineer", "AI Developer", "AI Programmer", "Applied Machine Learning Engineer",
        "Machine Learning Engineer", "Machine Learning Developer", "Machine Learning Infrastructure Engineer",
        "Machine Learning Software Engineer", "Machine Learning Scientist",
        "Machine Learning Researcher", "Machine Learning Research Engineer",
        "Lead Machine Learning Engineer", "Principal Machine Learning Engineer",
        "MLOps Engineer", "NLP Engineer"
    ],
    "Computer Vision / Deep Learning": [
        "3D Computer Vision Researcher", "Computer Vision Engineer", "Computer Vision Software Engineer",
        "Deep Learning Engineer", "Deep Learning Researcher", "Autonomous Vehicle Technician"
    ],
    "Data Leadership / Management": [
        "Data Manager", "Head of Data", "Head of Data Science", "Manager Data Management",
        "Data Lead", "Data Analytics Lead", "Data Science Manager", "Director of Data Science",
        "Lead Data Scientist", "Lead Data Engineer", "Principal Data Scientist", "Principal Data Engineer"
    ],
    "Data Architect / Infrastructure": [
        "Data Architect", "Big Data Architect", "Cloud Data Architect", "Cloud Database Engineer",
        "Data Modeler", "Principal Data Architect"
    ],
    "Data Analytics": [
        "Analytics Engineer", "Data Analytics Consultant", "Data Analytics Specialist"
    ],
    "Data Operations / Quality": [
        "Data Operations Analyst", "Data Operations Engineer", "Data Quality Analyst"
    ],
    "Data Governance / Strategy": [
        "Data Management Specialist", "Data Strategist", "Compliance Data Analyst"
    ],
    "Research Engineer": [
        "Research Engineer"
    ]
}


country_map = {
    'US': 'United States', 'GB': 'Great Britain', 'CA': 'Canada', 'ES': 'Spain',
    'IN': 'India', 'DE': 'Germany', 'FR': 'France', 'PT': 'Portugal', 'BR': 'Brazil',
    'GR': 'Greece', 'NL': 'Netherlands', 'AU': 'Australia', 'MX': 'Mexico',
    'IT': 'Italy', 'PK': 'Pakistan', 'JP': 'Japan', 'IE': 'Ireland', 'NG': 'Nigeria',
    'AT': 'Austria', 'AR': 'Argentina', 'PL': 'Poland', 'PR': 'Puerto Rico',
    'TR': 'Turkey', 'BE': 'Belgium', 'SG': 'Singapore', 'RU': 'Russia', 'LV': 'Latvia',
    'UA': 'Ukraine', 'CO': 'Colombia', 'CH': 'Switzerland', 'SI': 'Slovenia', 'BO': 'Bolivia',
    'DK': 'Denmark', 'HR': 'Croatia', 'HU': 'Hungary', 'RO': 'Romania', 'TH': 'Thailand',
    'AE': 'United Arab Emirates', 'VN': 'Vietnam', 'HK': 'Hong Kong',
    'UZ': 'Uzbekistan', 'PH': 'Philippines', 'CF': 'Central African Republic',
    'CL': 'Chile', 'FI': 'Finland', 'CZ': 'Czech Republic', 'SE': 'Sweden',
    'AS': 'American Samoa', 'LT': 'Lithuania', 'GH': 'Ghana', 'KE': 'Kenya',
    'DZ': 'Algeria', 'NZ': 'New Zeland', 'JE': 'Jersey', 'MY': 'Malaysia',
    'MD': 'Moldova', 'IQ': 'Iraq', 'BG': 'Bulgaria', 'LU': 'Luxembourg', 'RS': 'Serbia',
    'HN': 'Honduras', 'EE': 'Estonia', 'TN': 'Tunisia', 'CR': 'Costa Rica', 'ID': 'Indonesia',
    'EG': 'Egypt', 'DO': 'Dominican Republic', 'CN': 'China', 'SK': 'South Korea',
    'IR': 'Iran', 'MA': 'Morocco', 'IL': 'Israel', 'MK': 'North Macedonia', 'BA': 'Bosnia',
    'AM': 'Armenia', 'CY': 'Cyprus', 'KW': 'Kuwait', 'MT': 'Malta', 'BS': 'The Bahamas',
    'AL': 'Albania'
}


continent_map = {'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America', 'Puerto Rico': 'North America',
    'Honduras': 'North America', 'Costa Rica': 'North America',
    'Brazil': 'South America', 'Argentina': 'South America', 'Colombia': 'South America', 'Chile': 'South America', 'Bolivia': 'South America',
    'Great Britain': 'Europe', 'Germany': 'Europe', 'France': 'Europe', 'Greece': 'Europe',
    'Portugal': 'Europe', 'Netherlands': 'Europe', 'Ireland': 'Europe', 'Austria': 'Europe', 'Switzerland': 'Europe',
    'Poland': 'Europe', 'Latvia': 'Europe', 'Denmark': 'Europe', 'Italy': 'Europe', 'Slovenia': 'Europe', 'Belgium': 'Europe',
    'Ukraine': 'Europe', 'Croatia': 'Europe', 'Czech Republic': 'Europe', 'Finland': 'Europe', 'Luxembourg': 'Europe',
    'Hungary': 'Europe', 'Lithuania': 'Europe', 'Romania': 'Europe', 'Sweden': 'Europe', 'Estonia': 'Europe', 'Malta': 'Europe',
    'Russia': 'Europe', 'Turkey': 'Europe',
    'India': 'Asia', 'Japan': 'Asia', 'Singapore': 'Asia', 'Thailand': 'Asia', 'Indonesia': 'Asia',
    'China': 'Asia', 'Pakistan': 'Asia', 'South Korea': 'Asia', 'Vietnam': 'Asia', 'Philippines': 'Asia',
    'Malaysia': 'Asia', 'Hong Kong': 'Asia', 'Iran': 'Asia', 'Iraq': 'Asia', 'Armenia': 'Asia',
    'United Arab Emirates': 'Asia', 'Israel': 'Asia',
    'Egypt': 'Africa', 'Nigeria': 'Africa', 'Kenya': 'Africa', 'Ghana': 'Africa', 'Algeria': 'Africa',
    'Morocco': 'Africa', 'Central African Republic': 'Africa',
    'New Zeland': 'Oceania', 'Australia': 'Oceania', 'American Samoa': 'Oceania',
    'The Bahamas': 'North America', 'Moldova': 'Europe', 'Spain': 'Europe', 'Bosnia': 'Europe',
    'North Macedonia': 'Europe', 'Albania': 'Europe'}

ipc_by_year = {
    2020: 0.0136,
    2021: 0.0704,
    2022: 0.0645,
    2023: 0.0335,
    2024: 0.0289
}

adjustment_factors = {}
factor = 1.0
for year in range(2024, 2019, -1):
    factor = factor / (1 + ipc_by_year.get(year, 0))
    adjustment_factors[year] = 1 / factor

def transform_input(df: pd.DataFrame):
    def map_job(title):
        for general, specifics in job_mapping.items():
            if title in specifics:
                return general
        return "Other"

    #df['job_title'] = df['job_title'].apply(map_job)

    df.replace({
        'employment_type': {'FT': 'Full Time', 'PT': 'Part Time', 'CT': 'Contract', 'FL': 'Freelance'},
        'experience_level': {'SE': 'Senior', 'MI': 'Intermediate', 'EN': 'Entry', 'EX': 'Executive'},
        'company_size': {'S': 'Small', 'M': 'Medium', 'L': 'Large'},
        'employee_residence': country_map,
        'company_location': country_map
    }, inplace=True)

    df['continent'] = df['employee_residence'].map(continent_map)

    df["work_year"] = pd.to_datetime("today").year

    df["salary_in_usd"] = 0

    return df

def encode_input(df: pd.DataFrame):
    df_bin = binary_encoder.transform(df)
    df_temp = pd.concat([df.drop(columns=['employee_residence', 'company_location']), df_bin], axis=1)

    df_final = pd.get_dummies(df_temp)

    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    with open('column_order.json', 'r') as f:
        expected_columns = json.load(f)

    for col in expected_columns:
        if col not in df_final.columns:
            df_final[col] = 0

    df_final = df_final[expected_columns]

    return df_final


def predict_salary(professional_profile: ProfessionalProfile):
    df = pd.DataFrame([professional_profile.dict()])
    df = transform_input(df)
    df = encode_input(df)

    with open('reentrenado.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(df)
    return int(prediction[0])

@app.post("/predictions/TPOT/salaries",response_model=SalaryPredictionResponseDTO,description="Predict salary based on professional profile using TPOT model.", summary="Predict salary based on professional profile using TPOT model.")
async def create_salary_suggestion(professional_profile: ProfessionalProfile):
    print("Received professional profile:", professional_profile)
    suggested_salary = predict_salary(professional_profile)
    return {"sugested_salary": suggested_salary}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="api-salary-predictor",
        version="1.0.0",
        description="API for predicting salaries based on professional profiles using ML models.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    )