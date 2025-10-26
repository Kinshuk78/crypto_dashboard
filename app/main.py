# Import packages
import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os
from pathlib import Path

# Set Python path
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parents[0])
sys.path.append(parent_dir)

from students.solana_front import display_solana_front
from students.bitcoin_front import display_bitcoin_front
from students.ethereum_front import display_ethereum_front
from students.xrp_front import display_xrp_front

st.set_page_config(page_title="Crypto Dashboard", layout='wide', initial_sidebar_state='auto')

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Bitcoin", "Ethereum", "XRP", "Solana"],
        icons=["currency-bitcoin", "coin", "coin", "coin"],  # Bootstrap icons
        default_index=0,
        menu_icon="coins",  # optional: icon for the menu title
        orientation="vertical"
    )
# Display the selected page
if selected == "Bitcoin":
    display_bitcoin_front()
elif selected == "Ethereum":
    display_ethereum_front()
elif selected == "XRP":
    display_xrp_front()
elif selected == "Solana":
    display_solana_front()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 Powered by Adv MLAA AT3 - Group 29")
st.sidebar.markdown("🧑‍💻 Built with Streamlit")
