# legal_drafting.py - DRAFT Component: AI-Powered Legal Drafting

import re
import os

# CRITICAL: Set NO_PROXY before any network-related imports
os.environ['NO_PROXY'] = os.getenv('NO_PROXY', '192.168.0.56')

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import tiktoken

logger = logging.getLogger(__name__)
load_dotenv()

class LegalDraftingManager:
    """
    DRAFT Component - AI-Powered Legal Document Drafting
    Features:
    - Template-based document generation
    - Multiple document types and styles
    - Professional legal formatting
    - Clause library integration
    - Compliance and validation checks
    """
    
    def __init__(
        self,
        llm_model: str = "qwen2.5:14b",
        temperature: float = 0.3,
        max_tokens: int = 12000,
        user_preferences: Dict[str, str] = None
    ):
        """Initialize Legal Drafting Manager"""
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.user_preferences = user_preferences or {}
        
        # Initialize token counter
        try:
            self.token_counter = tiktoken.get_encoding("cl100k_base")
        except:
            self.token_counter = None
            logger.warning("tiktoken not available for token counting")
        
        # Initialize LLM
        self._initialize_llm()
        
        # Document templates
        self.document_templates = self._initialize_templates()
        
        # Clause library
        self.clause_library = self._initialize_clause_library()
        
        # Style guides
        self.style_guides = self._initialize_style_guides()
        
        logger.info("Legal Drafting Manager initialized")
    
    def _initialize_llm(self):
        """Initialize the legal drafting LLM with environment awareness"""
        try:
            app_env = os.getenv('APP_ENV', 'local')
            
            if app_env == 'production':
                logger.info("🚀 Initializing Groq Cloud LLM for DRAFT (Production Mode)")
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise ValueError("GROQ_API_KEY is required for production mode")
                
                self.llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=self.temperature,
                    groq_api_key=groq_api_key
                )
                logger.info("✅ Groq LLM initialized for Legal Drafting")
            else:
                logger.info("🏠 Initializing Ollama LLM for DRAFT (Local Mode)")
                base_url = os.getenv('OLLAMA_BASE_URL', 'http://192.168.0.56:11434')
                model = os.getenv('OLLAMA_MODEL', self.llm_model)
                
                self.llm = ChatOllama(
                    model=model,
                    temperature=self.temperature,
                    base_url=base_url,
                    timeout=180
                )
                logger.info(f"✅ Ollama LLM initialized: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize legal drafting LLM: {e}")
            raise
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize document templates"""
        return {
            "Contracts & Agreements": {
                "structure": [
                    "Title and Parties",
                    "Recitals",
                    "Definitions",
                    "Terms and Conditions",
                    "Performance Obligations", 
                    "Payment Terms",
                    "Representations and Warranties",
                    "Indemnification",
                    "Limitation of Liability",
                    "Term and Termination",
                    "Dispute Resolution",
                    "Governing Law",
                    "Miscellaneous Provisions",
                    "Signatures"
                ],
                "description": "Comprehensive commercial agreements and contracts"
            },
            "Petitions & Applications": {
                "structure": [
                    "Caption and Court Information",
                    "Parties and Jurisdiction",
                    "Statement of Facts",
                    "Legal Grounds",
                    "Prayer for Relief",
                    "Verification",
                    "Certificate of Service"
                ],
                "description": "Legal petitions and court applications"
            },
            "Court Orders & Judgments": {
                "structure": [
                    "Court and Case Information",
                    "Parties",
                    "Background and Findings",
                    "Legal Analysis",
                    "Orders and Directives",
                    "Effective Date and Compliance",
                    "Court Seal and Signature"
                ],
                "description": "Judicial orders and judgments"
            },
            "Legal Briefs & Submissions": {
                "structure": [
                    "Cover Page",
                    "Table of Contents",
                    "Table of Authorities",
                    "Statement of Issues",
                    "Statement of Facts",
                    "Argument",
                    "Conclusion",
                    "Certificate of Service"
                ],
                "description": "Legal briefs and court submissions"
            },
            "Statutes & Regulations": {
                "structure": [
                    "Title and Citation",
                    "Purpose and Scope",
                    "Definitions",
                    "Substantive Provisions",
                    "Enforcement Mechanisms",
                    "Penalties and Sanctions",
                    "Effective Date",
                    "Amendments and Repeals"
                ],
                "description": "Legislative and regulatory documents"
            }
        }
    
    def _initialize_clause_library(self) -> Dict[str, Dict[str, str]]:
        """Initialize standard legal clauses"""
        return {
            "Definitions": {
                "purpose": "Define key terms used throughout the document",
                "template": 'For purposes of this [Document Type], the following terms shall have the meanings set forth below:\n\n(a) "[Term]" means [definition];\n(b) "[Term]" means [definition];'
            },
            "Parties": {
                "purpose": "Identify the contracting parties",
                "template": 'This [Document Type] is entered into on [Date] between [Party 1 Name], a [Entity Type] organized under the laws of [Jurisdiction] ("[Party 1 Short Name]"), and [Party 2 Name], a [Entity Type] organized under the laws of [Jurisdiction] ("[Party 2 Short Name]").'
            },
            "Recitals": {
                "purpose": "Provide background and context",
                "template": 'WHEREAS, [background statement];\nWHEREAS, [background statement];\nNOW, THEREFORE, in consideration of the mutual covenants and agreements contained herein, the parties agree as follows:'
            },
            "Terms and Conditions": {
                "purpose": "Set forth primary obligations and rights",
                "template": '[Party] shall [obligation/right]. [Party] agrees to [specific commitment]. The terms of this agreement shall remain in effect for [duration].'
            },
            "Representations": {
                "purpose": "Statement of facts and assurances",
                "template": 'Each party represents and warrants that: (a) it has full corporate power and authority to enter into this Agreement; (b) the execution and performance of this Agreement has been duly authorized; (c) this Agreement constitutes a valid and binding obligation.'
            },
            "Warranties": {
                "purpose": "Guarantees and assurances of quality/performance",
                "template": '[Party] warrants that [warranty statement]. This warranty shall remain in effect for [duration] and covers [scope of warranty].'
            },
            "Limitations": {
                "purpose": "Limit liability and damages",
                "template": 'IN NO EVENT SHALL [PARTY] BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING WITHOUT LIMITATION LOST PROFITS, ARISING OUT OF OR RELATING TO THIS AGREEMENT.'
            },
            "Governing Law": {
                "purpose": "Specify applicable law and jurisdiction",
                "template": 'This Agreement shall be governed by and construed in accordance with the laws of [Jurisdiction], without regard to its conflict of laws principles. The parties consent to the exclusive jurisdiction of the courts of [Jurisdiction].'
            }
        }
    
    def _initialize_style_guides(self) -> Dict[str, Dict[str, str]]:
        """Initialize drafting style guides"""
        return {
            "Formal Legal": {
                "tone": "Traditional legal language with formal structure",
                "language": "Technical legal terminology, passive voice, complex sentences",
                "structure": "Hierarchical numbering, formal headings, traditional legal formatting"
            },
            "Plain English": {
                "tone": "Clear, accessible language while maintaining legal precision",
                "language": "Active voice, shorter sentences, minimal jargon with explanations",
                "structure": "Simple numbering, descriptive headings, user-friendly formatting"
            },
            "Technical": {
                "tone": "Precise technical language for complex legal matters",
                "language": "Specialized terminology, detailed definitions, comprehensive coverage",
                "structure": "Detailed cross-references, extensive definitions, technical precision"
            },
            "Conversational": {
                "tone": "Business-friendly language while maintaining legal validity",
                "language": "Natural language flow, business terminology, practical focus",
                "structure": "Bullet points, practical headings, business-oriented organization"
            }
        }
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.token_counter:
            try:
                return len(self.token_counter.encode(text))
            except:
                pass
        
        return int(len(text.split()) * 1.33)
    
    def _create_drafting_prompt(self, doc_type: str, style: str) -> PromptTemplate:
        """Create document-specific drafting prompt"""
        template = self._get_template_by_doc_type(doc_type)
        style_guide = self.style_guides.get(style, self.style_guides["Formal Legal"])
        jurisdiction = self.user_preferences.get('jurisdiction', 'General')
        
        structure_text = '\n'.join([f"- {section}" for section in template.get('structure', [])])
        
        prompt_template = f"""You are an expert legal document drafter specializing in {doc_type.lower()} with expertise in {jurisdiction} jurisdiction.

DRAFTING REQUIREMENTS:
- Document Type: {doc_type}
- Style: {style} - {style_guide['tone']}
- Language Guidelines: {style_guide['language']}
- Structure: {style_guide['structure']}
- Jurisdiction: {jurisdiction}

DOCUMENT STRUCTURE TO FOLLOW:
{structure_text}

DRAFTING GUIDELINES:
- Use professional legal language appropriate for {style.lower()} style
- Include all necessary legal elements and protections
- Ensure compliance with {jurisdiction} legal requirements
- Make provisions clear, enforceable, and comprehensive
- Include appropriate disclaimers and limitations
- Use proper legal formatting and organization
- Include placeholder text in [BRACKETS] for specific information that needs to be customized

USER REQUIREMENTS:
{{requirements}}

SPECIAL PROVISIONS REQUESTED:
{{special_provisions}}

CLAUSES TO INCLUDE:
{{clauses}}

DOCUMENT LENGTH GUIDANCE: {{length}}

Generate a complete, professional legal document that meets all requirements above. Include proper legal formatting, comprehensive provisions, and placeholder text where specific information needs to be inserted.

IMPORTANT: Start directly with the document content. Do not include any introductory text like "Here is..." or explanations before the document.

COMPLETE LEGAL DOCUMENT:"""

        return PromptTemplate(
            template=prompt_template,
            input_variables=["requirements", "special_provisions", "clauses", "length"]
        )
    
    def _get_template_by_doc_type(self, doc_type: str) -> Dict[str, Any]:
        """Get template structure by document type"""
        return self.document_templates.get(doc_type, {
            "structure": ["Title", "Body", "Conclusion", "Signatures"],
            "description": "General legal document"
        })
    
    def _format_clauses(self, clause_names: List[str]) -> str:
        """Format selected clauses for inclusion"""
        if not clause_names:
            return "Standard legal clauses as appropriate for document type"
        
        formatted_clauses = []
        for clause_name in clause_names:
            clause = self.clause_library.get(clause_name)
            if clause:
                formatted_clauses.append(f"- {clause_name}: {clause['purpose']}")
        
        return "\n".join(formatted_clauses) if formatted_clauses else "Standard legal clauses as appropriate for document type"
    
    def _determine_length_guidance(self, length: str) -> str:
        """Convert length selection to specific guidance"""
        length_guides = {
            "Brief": "2-4 pages, concise and focused, essential provisions only",
            "Standard": "5-10 pages, comprehensive standard provisions, typical commercial length",
            "Comprehensive": "10-20 pages, detailed provisions, extensive protections and clauses",
            "Detailed": "20+ pages, exhaustive coverage, complex multi-party arrangements"
        }
        return length_guides.get(length, "Standard length appropriate for document type")
    
    async def generate_document(
        self,
        doc_type: str,
        prompt: str,
        style: str = "Formal Legal",
        length: str = "Standard",
        clauses: List[str] = None,
        special_provisions: str = ""
    ) -> Dict[str, Any]:
        """Generate a legal document based on requirements"""
        start_time = time.time()
        
        try:
            if not prompt or not prompt.strip():
                return {
                    'document': 'Please provide requirements for the legal document you want to draft.',
                    'tokens_used': 0,
                    'error': 'No requirements provided'
                }
            
            logger.info(f"Generating {doc_type} in {style} style")
            
            # Prepare drafting inputs
            formatted_clauses = self._format_clauses(clauses or [])
            length_guidance = self._determine_length_guidance(length)
            
            # Create drafting prompt
            drafting_prompt = self._create_drafting_prompt(doc_type, style)
            
            formatted_prompt = drafting_prompt.format(
                requirements=prompt.strip(),
                special_provisions=special_provisions.strip() if special_provisions else "None specified",
                clauses=formatted_clauses,
                length=length_guidance
            )
            
            # Generate document
            logger.info("Generating legal document...")
            response = await self.llm.ainvoke(formatted_prompt)
            document = response.content if hasattr(response, 'content') else str(response)
            
            # Post-process document
            document = self._post_process_document(document, doc_type, style)
            
            # Calculate token usage
            input_tokens = self._count_tokens(formatted_prompt)
            output_tokens = self._count_tokens(document)
            total_tokens = input_tokens + output_tokens
            
            processing_time = time.time() - start_time
            
            logger.info(f"Document generation completed in {processing_time:.2f}s using {total_tokens} tokens")
            
            return {
                'document': document,
                'doc_type': doc_type,
                'style': style,
                'length': length,
                'clauses_included': clauses or [],
                'tokens_used': total_tokens,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'processing_time': processing_time,
                'word_count': len(document.split()),
                'metadata': {
                    'jurisdiction': self.user_preferences.get('jurisdiction', 'General'),
                    'generated_at': datetime.now().isoformat(),
                    'template_used': self._get_template_by_doc_type(doc_type)['description']
                }
            }
            
        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            return {
                'document': f'Document generation encountered an error: {str(e)}. Please try with different requirements or contact support.',
                'tokens_used': 0,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _post_process_document(self, document: str, doc_type: str, style: str) -> str:
        """Post-process generated document for formatting and quality"""
        try:
            # Remove any introductory phrases
            intro_patterns = [
                r'^.*?(?:Here is|Here\'s|I\'ve created|I\'ve generated|Below is).*?:\s*',
                r'^.*?(?:COMPLETE LEGAL DOCUMENT|LEGAL DOCUMENT).*?:\s*'
            ]
            
            for pattern in intro_patterns:
                document = re.sub(pattern, '', document, flags=re.IGNORECASE | re.DOTALL)
            
            # REMOVE ALL MARKDOWN SYNTAX - Make it clean!
            # Remove markdown headers (## Header -> Header)
            document = re.sub(r'^#{1,6}\s+', '', document, flags=re.MULTILINE)
            
            # Remove markdown bold (**text** -> text)
            document = re.sub(r'\*\*(.+?)\*\*', r'\1', document)
            
            # Remove markdown italics (*text* -> text)
            document = re.sub(r'\*(.+?)\*', r'\1', document)
            
            # Remove markdown bold with underscore (__text__ -> text)
            document = re.sub(r'__(.+?)__', r'\1', document)
            
            # Remove markdown italics with underscore (_text_ -> text)
            document = re.sub(r'_(.+?)_', r'\1', document)
            
            # Clean up formatting
            document = re.sub(r'\n\s*\n\s*\n+', '\n\n', document)  # Remove excessive line breaks
            document = re.sub(r'[ \t]+', ' ', document)  # Remove excessive whitespace
            
            # Ensure proper legal document structure
            if style == "Formal Legal":
                # Check if document has a proper title
                first_lines = document[:500]
                has_title = any(keyword in first_lines.upper() for keyword in 
                              ['AGREEMENT', 'CONTRACT', 'PETITION', 'ORDER', 'BRIEF', 'STATUTE', 'ACT'])
                
                if not has_title:
                    document_title = self._generate_document_title(doc_type)
                    document = f"{document_title}\n\n{document}"
            
            return document.strip()
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return document
    
    def _generate_document_title(self, doc_type: str) -> str:
        """Generate appropriate document title"""
        titles = {
            "Contracts & Agreements": "[DOCUMENT TITLE]\n\nAGREEMENT",
            "Petitions & Applications": "PETITION",
            "Court Orders & Judgments": "ORDER",
            "Legal Briefs & Submissions": "BRIEF IN SUPPORT OF [MOTION/POSITION]",
            "Statutes & Regulations": "[TITLE] ACT"
        }
        return titles.get(doc_type, f"{doc_type.upper()}")
    
    def get_available_templates(self) -> Dict[str, Any]:
        """Get information about available document templates"""
        templates_info = {}
        
        for doc_type, template in self.document_templates.items():
            templates_info[doc_type] = {
                'description': template['description'],
                'structure_elements': len(template['structure']),
                'typical_sections': template['structure'][:5]  # Show first 5 sections
            }
        
        return {
            'available_templates': templates_info,
            'total_templates': len(self.document_templates),
            'supported_styles': list(self.style_guides.keys()),
            'available_clauses': list(self.clause_library.keys())
        }
    
    def get_clause_details(self, clause_name: str) -> Dict[str, Any]:
        """Get details about a specific clause"""
        clause = self.clause_library.get(clause_name)
        
        if clause:
            return {
                'name': clause_name,
                'purpose': clause['purpose'],
                'template': clause['template'],
                'usage_notes': f"Standard {clause_name.lower()} clause for legal documents"
            }
        
        return {
            'error': f'Clause "{clause_name}" not found',
            'available_clauses': list(self.clause_library.keys())
        }
    
    def validate_document_requirements(self, requirements: str, doc_type: str) -> Dict[str, Any]:
        """Validate document requirements for completeness"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'suggestions': [],
            'missing_elements': []
        }
        
        # Check for essential elements based on document type
        essential_elements = {
            "Contracts & Agreements": ['parties', 'consideration', 'terms', 'obligations'],
            "Petitions & Applications": ['petitioner', 'respondent', 'relief sought', 'grounds'],
            "Legal Briefs & Submissions": ['argument', 'legal authority', 'facts', 'conclusion']
        }
        
        required_elements = essential_elements.get(doc_type, [])
        requirements_lower = requirements.lower()
        
        for element in required_elements:
            if element not in requirements_lower:
                validation_results['missing_elements'].append(element)
                validation_results['suggestions'].append(f"Consider specifying {element} in your requirements")
        
        # Check length and detail level
        if len(requirements.split()) < 20:
            validation_results['warnings'].append("Requirements appear brief - more detail may improve document quality")
        
        # Check for specific legal terms that might need clarification
        legal_terms = ['liability', 'indemnification', 'breach', 'termination', 'governing law']
        mentioned_terms = [term for term in legal_terms if term in requirements_lower]
        
        if mentioned_terms:
            validation_results['suggestions'].append(f"Consider providing specific details about: {', '.join(mentioned_terms)}")
        
        return validation_results
    
    def update_templates(self, new_templates: Dict[str, Any]):
        """Update document templates (for advanced users)"""
        for doc_type, template in new_templates.items():
            if 'structure' in template and 'description' in template:
                self.document_templates[doc_type] = template
                logger.info(f"Updated template for {doc_type}")
            else:
                logger.warning(f"Invalid template format for {doc_type}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about document generation"""
        return {
            'model': self.llm_model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'available_templates': len(self.document_templates),
            'available_clauses': len(self.clause_library),
            'supported_styles': len(self.style_guides),
            'jurisdiction': self.user_preferences.get('jurisdiction', 'General')
        }