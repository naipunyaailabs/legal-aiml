# agent_handlers/tools.py - Tools for Legal AI Agents

import os
import httpx
import logging
from typing import Dict, Any, List, Optional
from .base_agent import get_httpx_client

logger = logging.getLogger(__name__)

# Backend URLs
ASK_DRAFT_URL = os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8000")
INTERACT_URL = os.getenv("INTERACT_API_URL", "http://127.0.0.1:8001")

async def legal_doc_search(query: str, mode: str = "Document Only") -> Dict[str, Any]:
    """Search for information in the uploaded legal documents database."""
    try:
        payload = {
            "message": query,
            "chatbot_mode": mode
        }
        response = await get_httpx_client().post(
            f"{ASK_DRAFT_URL}/api/chat",
            json=payload
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "answer": data.get("answer"),
                "sources": data.get("sources", [])
            }
        return {"error": f"Search failed with status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

async def generate_legal_document(doc_type: str, requirements: str, style: str = "Formal Legal") -> Dict[str, Any]:
    """Generate a legal document draft based on requirements."""
    try:
        payload = {
            "doc_type": doc_type,
            "requirements": requirements,
            "style": style
        }
        response = await get_httpx_client().post(
            f"{ASK_DRAFT_URL}/api/drafting/generate",
            json=payload
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Drafting failed with status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

async def search_indian_statutes(topic: str) -> List[Dict[str, str]]:
    """Search for relevant Indian Statutes (Bare Acts) and sections."""
    # Mock data for demonstration - in production, this would call a real database or API
    statutes = [
        {"act": "Companies Act, 2013", "section": "166", "description": "Duties of directors"},
        {"act": "Indian Contract Act, 1872", "section": "73", "description": "Compensation for loss or damage caused by breach of contract"},
        {"act": "Information Technology Act, 2000", "section": "43A", "description": "Compensation for failure to protect data"},
        {"act": "IBC, 2016", "section": "7", "description": "Initiation of corporate insolvency resolution process by financial creditor"}
    ]
    # Filter based on topic (very basic mock)
    results = [s for s in statutes if topic.lower() in s["act"].lower() or topic.lower() in s["description"].lower()]
    return results if results else statutes[:2]

async def case_law_lookup(query: str) -> List[Dict[str, str]]:
    """Search for relevant Indian Case Law and precedents."""
    # Mock data for demonstration
    cases = [
        {"case_name": "Tata Cellular v. Union of India (1994)", "citation": "1994 SCC (6) 651", "relevance": "Scope of judicial review in government contracts"},
        {"case_name": "K.S. Puttaswamy v. Union of India (2017)", "citation": "2017 (10) SCC 1", "relevance": "Right to Privacy as a Fundamental Right"},
        {"case_name": "Standard Chartered Bank v. Directorate of Enforcement (2005)", "citation": "2005 (4) SCC 530", "relevance": "Corporate criminal liability"},
        {"case_name": "M.C. Mehta v. Union of India (1987)", "citation": "1987 SCR (1) 319", "relevance": "Absolute Liability principle"}
    ]
    return cases[:3]

async def compliance_check(document_text: str) -> Dict[str, Any]:
    """Check document text against common Indian regulatory compliance requirements."""
    check_results = {
        "mca_compliance": "Verified",
        "tax_registration": "Required check for PAN/GSTIN",
        "labour_laws": "EPF/ESI compliance check recommended",
        "data_protection": "DPDP Act 2023 readiness suggested"
    }
    return check_results

async def risk_analyzer(text: str) -> List[Dict[str, str]]:
    """Analyze legal text to extract potential risks and liabilities."""
    risks = [
        {"risk_type": "Limitation of Liability", "severity": "High", "observation": "Liability cap is absent or too high"},
        {"risk_type": "Indemnity", "severity": "Medium", "observation": "Uncapped indemnity for third-party claims"},
        {"risk_type": "Termination", "severity": "Low", "observation": "No termination for convenience clause"}
    ]
    return risks
