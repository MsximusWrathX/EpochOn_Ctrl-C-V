import os
from typing import List, Dict, Callable
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

class ProsecutorAgent:
    def __init__(self, gemini_api_key: str, tavily_api_key: str, status_callback: Callable[[str], None] = None):
        # Using Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=gemini_api_key,
            temperature=0.6 # Higher temperature for creative prosecution
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.status_callback = status_callback
        
    def find_legal_precedents(self, feature: str):
        """Search for legal precedents or regulations that might be violated."""
        query = f"legal risks and failure cases of {feature} in courthouses"
        response = self.tavily_client.search(query=query, search_depth="advanced", max_results=3)
        return response.get('results', [])

    def prosecute_model(self, model_description: str, defense_arguments: str = None) -> str:
        """The Prosecutor's core logic: attacking the courthouse design."""
        if self.status_callback:
            self.status_callback("⚖️ The Prosecution is preparing the indictment...")
        
        # Optional: Search for damaging data
        damage_data = self.find_legal_precedents("modern open-plan") 
        damage_context = "\n".join([f"- {s['content']}" for s in damage_data])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Chief Prosecutor representing the Public Interest and Judicial Safety standards.
            Your goal is to relentlessly cross-examine the proposed courthouse model and expose every flaw, safety risk, and inefficiency.

            Your responsibilities:
            1. Opening Statement: Declare the model "guilty" of poor design, safety violations, or wastefulness.
            2. Cross-Examination: If the Defense has spoken, tear apart their arguments. Point out contradictions or idealism that ignores reality.
            3. Cite Violations: Use the provided context (Exhibit B) to show where similar designs have failed or are illegal.
            4. Closing Argument: Demand that the model be rejected (convicted) for the safety of the public.
            5. Tone: Accusatory, sharp, authoritative, and uncompromising. Use legal terms like "negligence", "liability", "motion to strike", "objection"."""),
            ("user", f"""
            EVIDENCE OF FAILURES (EXHIBIT B):
            {damage_context}

            DEFENDANT'S PROPOSED MODEL:
            {model_description}

            DEFENSE ARGUMENTS (IF ANY):
            {defense_arguments if defense_arguments else "The Defense has remained silent."}

            Prosecute this model immediately:""")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({})
            return response.content
        except Exception as e:
            return f"❌ Prosecution error: {str(e)}"

    def prosecute(self, model_data: str, defense: str = None):
        """Main entry point for the agent."""
        if self.status_callback:
            self.status_callback("📜 Prosecutor is filing charges...")
        
        result = self.prosecute_model(model_data, defense)
        
        if self.status_callback:
            self.status_callback("✅ Indictment filed.")
            
        return result

class ProsecutionStrategistAgent:
    def __init__(self, groq_api_key: str, tavily_api_key: str, status_callback: Callable[[str], None] = None):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            groq_api_key=groq_api_key,
            temperature=0.3 # Sharp, factual, and ruthless
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.status_callback = status_callback
        
    def find_counter_evidence(self, defense_claim: str):
        """Search for evidence that disproves or weakens the defense's claim."""
        query = f"evidence against {defense_claim} and failures in construction"
        response = self.tavily_client.search(query=query, search_depth="advanced", max_results=3)
        return response.get('results', [])

    def shred_defense(self, model_description: str, defense_argument: str) -> str:
        """The Strategist's core logic: Dismantling the Defense's case."""
        if self.status_callback:
            self.status_callback("🕵️ Prosecution Strategist is reviewing the Defense's lies...")
        
        # Step 1: Find weakness in Defense
        # Assume Defense praises "innovation". Search for failures of that innovation.
        rebuttal_evidence = self.find_counter_evidence("unproven architectural innovations")
        rebuttal_context = "\n".join([f"- {s['content']}" for s in rebuttal_evidence])

        # Step 2: Ruthless Strategic Analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Chief Prosecution Strategist.
            Your ONLY job is to destroy the credibility of the Defense Attorney.
            You are NOT judging the model. You are attacking the Defense's Argument.
            
            Evaluate the Defense's case based on:
            1. Emotional Manipulation: Is the defense using sob stories instead of facts?
            2. Safety Violations: Does the defense ignore public safety regulations?
            3. Cost: Is the defense hiding the true cost to the taxpayer?
            4. Attack Plan: Provide 3 lethal questions the Prosecutor (Advocate B) should ask on cross-examination.
            
            Be ruthless, precise, and completely intolerant of vague "visionary" talk."""),
            ("user", f"""
            DAMNING EVIDENCE (REBUTTAL):
            {rebuttal_context}

            DEFENDANT'S MODEL:
            {model_description}

            DEFENSE ARGUMENT:
            {defense_argument}

            Provide a plan to crush the defense:""")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({})
            return response.content
        except Exception as e:
            return f"❌ Strategy error: {str(e)}"

    def strategize(self, model_data: str, defense_arg: str):
        """Main entry point for the agent."""
        if self.status_callback:
            self.status_callback("🧠 Formulating prosecution strategy...")
        
        result = self.shred_defense(model_data, defense_arg)
        
        if self.status_callback:
            self.status_callback("✅ Attack plan ready.")
            
        return result
