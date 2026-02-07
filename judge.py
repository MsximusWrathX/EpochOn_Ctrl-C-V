import os
import json
from typing import List, Dict, Callable
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

class JudgeAgent:
    def __init__(self, groq_api_key: str, tavily_api_key: str, status_callback: Callable[[str], None] = None):
        """
        Initializes the Judge Agent.
        
        Args:
            groq_api_key: API key for Groq (Llama-3).
            tavily_api_key: API key for Tavily search.
            status_callback: Optional callback for status updates.
        """
        self.llm = ChatGroq(
            temperature=0, # Zero temperature for maximum objectivity
            model="llama-3.3-70b-versatile", # High reasoning capability
            groq_api_key=groq_api_key
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.status_callback = status_callback

    def verify_key_claims(self, claims: List[str]) -> List[Dict]:
        """Fact-checks specific claims using Tavily."""
        verifications = []
        for claim in claims:
            if self.status_callback:
                self.status_callback(f"🔍 Judge is verifying claim: '{claim[:50]}...'")
            
            # Focused search for verification
            query = f"fact check {claim} true or false evidence"
            try:
                search_result = self.tavily_client.search(query=query, search_depth="advanced", max_results=2)
                verifications.append({
                    "claim": claim,
                    "evidence": [r['content'] for r in search_result.get('results', [])]
                })
            except Exception as e:
                verifications.append({"claim": claim, "error": str(e)})
        return verifications

    def deliberate(self, 
                   defense_brief: str, 
                   prosecution_brief: str, 
                   defense_strategy: str, 
                   prosecution_strategy: str) -> str:
        """
        The Judge's main logic loop:
        1. Parse arguments.
        2. Identify disputed facts.
        3. Independently verify critical claims via Tavily.
        4. Evaluate consensus and confidence.
        5. Render a verdict or Refuse to Decide.
        """
        if self.status_callback:
            self.status_callback("🧑‍⚖️ The Court is now in session. Reviewing all briefs...")

        # Step 1: Synthesize and Identify Claims to Check
        # We ask Groq to extract 3 critical fact-based claims that conflict.
        claim_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Judicial Clerk. Extract 3 specific, verifiable factual claims that are in dispute between the Defense and Prosecution."),
            ("user", f"""
            DEFENSE BRIEF: {defense_brief}
            PROSECUTION BRIEF: {prosecution_brief}
            
            Return ONLY a JSON list of strings, e.g. ["claim 1", "claim 2"]
            """)
        ])
        
        try:
            extraction_chain = claim_extraction_prompt | self.llm
            claims_json = extraction_chain.invoke({}).content
            # Cleanup json string if needed
            if "```json" in claims_json:
                claims_json = claims_json.split("```json")[1].split("```")[0]
            claims_to_check = json.loads(claims_json)
        except:
            claims_to_check = ["safety compliance", "cost efficiency", "historical precedent"] # Fallback

        # Step 2: Verification
        verification_results = self.verify_key_claims(claims_to_check)
        verification_text = json.dumps(verification_results, indent=2)

        if self.status_callback:
            self.status_callback("⚖️ Deliberating on the findings...")

        # Step 3: Final Judgment
        judgment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Supreme Judge of Architectural Law.
            Your duty is to render a verdict based ONLY on facts and safety.
            
            CRITICAL RULES:
            1. CONSENSUS CHECK: If the independent verification (Tavily) contradicts both sides or is inconclusive on safety critical issues, you MUST REFUSE to decide.
            2. SAFETY FIRST: Any confirmed safety violation is immediate grounds for ruling against the model.
            3. OBJECTIVITY: Ignore emotional appeals from the Defense or Prosecution.
            
            Output your report in the following MarkDown format:
            
            ## 🏛️ Judicial Report
            
            ### 🛡️ Defense Summary
            [Brief Summary]
            
            ### ⚖️ Prosecution Summary
            [Brief Summary]
            
            ### 🔍 Independent Verification (Tavily Core)
            [Summarize the Tavily findings for the disputed claims]
            
            ### 📊 Confidence Analysis
            - **Confidence Score**: [0-100]%
            - **Consensus Status**: [Strong/Weak/Conflict]
            
            ### 🧑‍⚖️ Final Decision
            [VERDICT: DEFENSE WINS | VERDICT: PROSECUTION WINS | REFUSAL: NEW SESSION ORDERED]
            
            [Detailed reasoning for the decision/refusal]"""),
            ("user", f"""
            DEFENSE ARGUMENTS: {defense_brief}
            PROSECUTION ARGUMENTS: {prosecution_brief}
            DEFENSE STRATEGY: {defense_strategy}
            PROSECUTION STRATEGY: {prosecution_strategy}
            
            INDEPENDENT FACT CHECK RESULTS (TAVILY):
            {verification_text}
            
            Render your decision now:""")
        ])

        judgment_chain = judgment_prompt | self.llm
        verdict = judgment_chain.invoke({}).content
        
        if self.status_callback:
            self.status_callback("✅ The Judge has reached a decision.")
            
        return verdict
