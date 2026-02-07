import streamlit as st
import time
import os
from dotenv import load_dotenv

# Import the actual agent teams
from defense_team import DefenseAttorneyAgent, DefenseStrategistAgent
from prosecution_team import ProsecutorAgent, ProsecutionStrategistAgent
from judge import JudgeAgent
from utils import generate_content # Keep for CaseManager initial summary

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="AI Legal Debate Simulation", layout="wide", page_icon="⚖️")

# --- INITIALIZATION ---
def get_api_keys():
    """Retrieve API keys from environment variables."""
    keys = {
        "gemini_1": os.getenv("GEMINI_API_KEY1"), # Defense Attorney
        "gemini_2": os.getenv("GEMINI_API_KEY2"), # Prosecutor
        "groq_1": os.getenv("GROQ_API_KEY1"),    # Defense Strategist
        "groq_2": os.getenv("GROQ_API_KEY2"),    # Prosecution Strategist
        "groq_3": os.getenv("GROQ_API_KEY3"),    # Judge
        "tavily": os.getenv("TAVILY_API_KEY")    # Judge & Teams
    }
    
    # Check for missing keys
    missing = [k for k, v in keys.items() if not v]
    if missing:
        st.error(f"Missing API Keys in .env: {', '.join(missing)}")
        st.stop()
    return keys

# Initialize Agents
@st.cache_resource
def initialize_agents():
    keys = get_api_keys()
    
    # st.toast("Initializing Legal Teams...", icon="⚖️") # Removed to fix CacheReplayClosureError
    
    # Defense Team (Gemini for Advocate, Groq for Strategist)
    defense_attorney = DefenseAttorneyAgent(gemini_api_key=keys["gemini_1"], tavily_api_key=keys["tavily"])
    defense_strategist = DefenseStrategistAgent(groq_api_key=keys["groq_1"], tavily_api_key=keys["tavily"])
    
    # Prosecution Team (Gemini for Prosecutor, Groq for Strategist)
    prosecutor = ProsecutorAgent(gemini_api_key=keys["gemini_2"], tavily_api_key=keys["tavily"])
    prosecution_strategist = ProsecutionStrategistAgent(groq_api_key=keys["groq_2"], tavily_api_key=keys["tavily"])
    
    # Judge (Groq + Tavily)
    judge = JudgeAgent(groq_api_key=keys["groq_3"], tavily_api_key=keys["tavily"])
    
    return defense_attorney, defense_strategist, prosecutor, prosecution_strategist, judge

# Helper for Case Summary (using simple utility function)
class CaseManager:
    def __init__(self, case_description):
        self.case_description = case_description

    def summarize_case(self):
        prompt = f"""
        Analyze the following legal case description and extract key facts. 
        Provide a structured summary suitable for a legal debate.
        
        Case Description:
        {self.case_description}
        """
        # Fallback to utils.generate_content (which uses a default key/model)
        # Ideally this should also use one of the specific keys, but keeping as is for now
        # assuming utils.py is configured correctly.
        return generate_content(prompt)

# --- MAIN INTERFACE ---
st.title("⚖️ AI Courtroom: Prosecution vs Defense")
st.markdown("### Agentic Workflow with Strategists & Judicial Oversight")

# Sidebar Configuration
st.sidebar.title("Configuration")
num_rounds = st.sidebar.slider("Number of Rounds", 1, 3, 1)

# Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "case_summary" not in st.session_state:
    st.session_state.case_summary = None
if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = False

# Input Area
case_input = st.text_area("Enter Case Details / Facts", height=150, placeholder="Describe the legal case, crime, or dispute here...")

if st.button("Start Court Session", type="primary"):
    if not case_input:
        st.warning("Please enter a case description.")
    else:
        st.session_state.run_simulation = True
        with st.spinner("Clerk is summarizing the case..."):
            case_manager = CaseManager(case_input)
            summary = case_manager.summarize_case()
            st.session_state.case_summary = summary
            st.session_state.history = []

if st.session_state.run_simulation and st.session_state.case_summary:
    st.success("Case Docket Created")
    with st.expander("View Case Summary", expanded=True):
        st.write(st.session_state.case_summary)
    
    # Load Agents
    defense_attorney, defense_strategist, prosecutor, prosecution_strategist, judge = initialize_agents()
    
    defense_brief = ""
    prosecution_brief = ""
    defense_strategy_doc = ""
    prosecution_strategy_doc = ""

    # --- SIMULATION LOOP ---
    for r in range(num_rounds):
        st.markdown("---")
        st.subheader(f"Session Round {r+1}")
        
        col1, col2 = st.columns(2)
        
        # 1. Prosecution Case (Prosecutor always starts criminal cases)
        with col1:
            st.markdown("### 🏛️ Prosecution")
            with st.spinner("Prosecution Team is strategizing..."):
                # Strategy Step
                if r == 0:
                    p_strat = prosecution_strategist.strategize(st.session_state.case_summary, "Initial Opening Strategy")
                else:
                    p_strat = prosecution_strategist.strategize(st.session_state.case_summary, defense_brief)
                
                prosecution_strategy_doc += f"\nRound {r+1}: {p_strat}\n"
                with st.expander("View Prosecution Strategy (Internal)", expanded=False):
                    st.info(p_strat)
            
            with st.spinner("Prosecutor is presenting argument..."):
                # Argument Step
                if r == 0:
                     p_arg = prosecutor.prosecute(st.session_state.case_summary, "Opening Statement")
                else:
                     p_arg = prosecutor.prosecute(st.session_state.case_summary, defense_brief)
                
                prosecution_brief += f"\nRound {r+1}: {p_arg}\n"
                st.chat_message("assistant", avatar="⚖️").write(p_arg)

        # 2. Defense Case
        with col2:
            st.markdown("### 🛡️ Defense")
            with st.spinner("Defense Team is analyzing..."):
                # Strategy Step
                d_strat = defense_strategist.strategize(st.session_state.case_summary, p_arg)
                defense_strategy_doc += f"\nRound {r+1}: {d_strat}\n"
                with st.expander("View Defense Strategy (Internal)", expanded=False):
                    st.info(d_strat)
            
            with st.spinner("Defense Attorney is rebutting..."):
                # Argument Step
                d_arg = defense_attorney.advocate(st.session_state.case_summary, p_arg)
                defense_brief += f"\nRound {r+1}: {d_arg}\n"
                st.chat_message("user", avatar="🛡️").write(d_arg)

        # 3. Judicial Note
        st.caption(f"End of Round {r+1}. The Judge is taking notes.")

    # --- FINAL VERDICT ---
    st.markdown("---")
    st.header("🧑‍⚖️ Final Verdict")
    
    if st.button("Request Verdict"):
        with st.spinner("The Judge is deliberating (checking facts with Tavily)..."):
            verdict = judge.deliberate(
                defense_brief=defense_brief,
                prosecution_brief=prosecution_brief,
                defense_strategy=defense_strategy_doc,
                prosecution_strategy=prosecution_strategy_doc
            )
            st.markdown(verdict)
