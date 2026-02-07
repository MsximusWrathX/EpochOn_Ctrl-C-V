import os
from typing import List, Dict, Callable
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

class DefenseAttorneyAgent:
    def __init__(self, gemini_api_key: str, tavily_api_key: str, status_callback: Callable[[str], None] = None):
        # Using Gemini (Default Model)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            google_api_key=gemini_api_key,
            temperature=0.6 # Higher temperature allows for more "creative" justification
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.status_callback = status_callback
        
    def find_supporting_precedents(self, feature: str):
        """Search for successful examples or legal precedents that support a feature."""
        query = f"successful examples and benefits of {feature} in modern courthouses"
        response = self.tavily_client.search(query=query, search_depth="advanced", max_results=3)
        return response.get('results', [])

    def defend_model(self, model_description: str, critique_points: str = None) -> str:
        """The Advocate's core logic: Defending the courthouse design."""
        if self.status_callback:
            self.status_callback("🛡️ The Defense is gathering evidence and precedents...")
        
        # Optional: Search for supporting data based on the model description
        support_data = self.find_supporting_precedents("modern open-plan") # Example feature
        support_context = "\n".join([f"- {s['content']}" for s in support_data])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a lead Defense Attorney specializing in Architectural Liability and Construction Law.
            Your client is the Architect of the proposed courthouse model. Your goal is to zealously defend the design against all charges of inefficiency or flaw.

            Your responsibilities:
            1. Opening Statement: Present the model's design choices as deliberate, lawful, and necessary for justice.
            2. Rebuttal / Cross-Examination: If opposing counsel (the Critic) presents flaws, argue that these concerns are speculative, inadmissible, or outweighed by the benefits. Cast "reasonable doubt" on the validity of the critique.
            3. Cite Precedents: Reference successful examples (Exhibits A, B, etc.) to prove the design is sound.
            4. Closing Argument: Summarize why the model must be accepted (acquitted) as the superior choice.
            5. Tone: Use legal terminology (e.g., "submit into evidence", "objection", "precedent", "your Honor", "client's intent"), be persuasive, protective of the client, and professional."""),
            ("user", f"""
            EVIDENCE / PRECEDENTS (EXHIBITS):
            {support_context}

            CLIENT'S PROPOSED MODEL:
            {model_description}

            PROSECUTION'S CRITIQUE (IF ANY):
            {critique_points if critique_points else "No charges filed yet."}

            Provide a compelling legal defense of this model:""")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({})
            return response.content
        except Exception as e:
            return f"❌ Defense error: {str(e)}"

    def advocate(self, model_data: str, critique: str = None):
        """Main entry point for the agent."""
        if self.status_callback:
            self.status_callback("⚖️ Defense Attorney is preparing the closing statement...")
        
        result = self.defend_model(model_data, critique)
        
        if self.status_callback:
            self.status_callback("✅ Closing arguments ready.")
            
        return result

class DefenseStrategistAgent:
    def __init__(self, groq_api_key: str, tavily_api_key: str, status_callback: Callable[[str], None] = None):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            groq_api_key=groq_api_key,
            temperature=0.4 # Slightly creative to find unique angles
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.status_callback = status_callback
        
    def find_legal_loopholes(self, prosecutor_point: str):
        """Search for legal precedents that contradict the prosecutor's claims."""
        query = f"legal exceptions and successful defenses against {prosecutor_point} in construction law"
        response = self.tavily_client.search(query=query, search_depth="advanced", max_results=3)
        return response.get('results', [])

    def dismantle_prosecution(self, model_description: str, prosecutor_argument: str) -> str:
        """The Strategist's core logic: Destroying the Prosecutor's case."""
        if self.status_callback:
            self.status_callback("🕵️ Defense Strategist is analyzing the Prosecution's case...")
        
        # Step 1: Find counter-evidence against key prosecution points
        # Keep it simple: Assume the prosecutor attacks safety.
        counter_evidence = self.find_legal_loopholes("strict liability in design")
        loophole_context = "\n".join([f"- {s['content']}" for s in counter_evidence])

        # Step 2: Strategic Analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Legal Strategist for the Defense.
            Your ONLY job is to find the flaws, gaps, and legal errors in the Prosecutor's argument.
            You are NOT judging the model. You are attacking the Prosecutor's logic.
            
            Evaluate the Prosecutor's case based on:
            1. Logical Fallacies: Is the prosecutor using slippery slope or ad hominem attacks?
            2. Lack of Precedent: Is their argument purely speculative?
            3. Misinterpretation: Have they misunderstood the design intent?
            4. Counter-Strategy: Provide 3 specific legal arguments the Defense Lawyer (Advocate A) should use in rebuttal.
            
            Be sharp, cynical, and 100% on the side of the Defense."""),
            ("user", f"""
            LEGAL LOOPHOLES & PRECEDENTS:
            {loophole_context}

            DEFENDANT'S MODEL:
            {model_description}

            PROSECUTOR'S ARGUMENT:
            {prosecutor_argument}

            Provide a strategic breakdown of the prosecution's weaknesses:""")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({})
            return response.content
        except Exception as e:
            return f"❌ Strategy error: {str(e)}"

    def strategize(self, model_data: str, prosecutor_arg: str):
        """Main entry point for the agent."""
        if self.status_callback:
            self.status_callback("🧠 Formulating defense strategy...")
        
        result = self.dismantle_prosecution(model_data, prosecutor_arg)
        
        if self.status_callback:
            self.status_callback("✅ Strategy briefing ready.")
            
        return result
