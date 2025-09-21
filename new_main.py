import streamlit as st
import requests
import time

LANGUAGE_MAP = {
  "English": "en",
  "Hindi": "hi",
  "Marathi": "mr"
}
st.set_page_config(page_title=" Chatbot", layout="centered")
st.title("🤖 Policy Chatbot")

# Custom CSS for better chat interface
# Add this at the top of your main.py (right after st.set_page_config)
st.markdown("""
<style>
/* Main chat container */
.chat-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 200px);
  padding-bottom: 80px; /* Space for input */
}

/* Chat history area */
.chat-history {
  flex-grow: 1;
  overflow-y: auto;
  margin-bottom: 20px;
...#UI
}
</style>
""", unsafe_allow_html=True)

# Session state
if "authenticated" not in st.session_state:
  st.session_state.authenticated = False
  st.session_state.role = None
  st.session_state.username = None
  st.session_state.conversation_history = []
  st.session_state.session_id = None
  st.session_state.faq_suggestions = []
  st.session_state.file_uploaded = False
  st.session_state.selected_language = "English", "Hindi", "Marathi" # Default language
  st.session_state.clear_input = False

def start_new_conversation():
  try:
    res = requests.post(
      "http://localhost:8000/start-conversation",
      json={"username": st.session_state.username,
         "language": LANGUAGE_MAP[language]}
    )
    if res.status_code == 200:
      st.session_state.session_id = res.json().get("session_id")
      st.session_state.language = LANGUAGE_MAP[language]
      return True
    st.error("Failed to start conversation")
    return False
  except Exception as e:
    st.error(f"Error: {str(e)}")
    return False

# Function to display references
def display_references(references, message_index):
  if references:
    with st.expander(f"📚 References for Answer {message_index+1}", expanded=False):
      for ref_idx, ref in enumerate(references):
        col1, col2 = st.columns([4, 1])
        with col1:
          st.markdown(f"""
          **Document:** {ref['document']} 
          **Excerpt:** {ref['content']}
          """)
        with col2:
          doc_name = ref['document']
          download_url = f"http://localhost:8000/download/{doc_name}"
          st.download_button(
            label="📥 Download",
            data=requests.get(download_url).content,
            file_name=doc_name,
            mime="application/octet-stream",
            key=f"dl_{message_index}_{ref_idx}"
          )
        st.divider()
# Login form
if not st.session_state.authenticated:
  st.markdown("Login to ask questions about company policies.")
  with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    language = st.selectbox(
      "Preferred Language",
      list(LANGUAGE_MAP.keys()),
      index=list(LANGUAGE_MAP.values()).index(st.session_state.get('language', 'en'))
    )
    submitted = st.form_submit_button("Login")

    if submitted:
      with st.spinner("Authenticating..."):
        try:
          res = requests.post(
            "http://localhost:8000/login",
            json={"username": username, "password": password}
          )
          if res.status_code == 200:
            data = res.json()
            st.session_state.authenticated = True
            st.session_state.role = data["role"]
            st.session_state.username = username
            st.session_state.language = LANGUAGE_MAP[language]

            if not start_new_conversation():
              st.error("Failed to initialize conversation")
              st.session_state.authenticated = False
            else:
              st.success("Login successful!")
              time.sleep(1)
              st.rerun()
          else:
            st.error("Invalid credentials")
        except requests.exceptions.ConnectionError:
          st.error("Could not connect to server")
        except Exception as e:
          st.error(f"Error: {str(e)}")

# Main interface
if st.session_state.authenticated:
  # Header with user info and logout
  col1, col2 = st.columns([4, 1])
  with col1:
    st.markdown(f"Welcome, *{st.session_state.username}* (Role: {st.session_state.role})**Language:** {list(LANGUAGE_MAP.keys())[list(LANGUAGE_MAP.values()).index(st.session_state.language)]}")
  with col2:
    if st.button("Logout"):
      try:
        requests.post("http://localhost:8000/logout", json={"session_id": st.session_state.session_id})
      except:
        pass # fail silently
      st.session_state.authenticated = False
      st.session_state.role = None
      st.session_state.username = None
      st.session_state.conversation_history = []
      st.session_state.session_id = None
      st.session_state.faq_suggestions = []
      st.rerun()

  # Admin document upload section
  if st.session_state.role == "admin":
    with st.expander("📤 Document Upload", expanded=False):
      tab1, tab2 = st.tabs(["Single Upload", "Bulk Upload"])
      with tab1:
        uploaded_file = st.file_uploader("Upload a single document", type=["pdf", "docx", "csv"])
        if uploaded_file is not None:
          with st.spinner("Uploading..."):
            try:
              files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
              data = {"username": st.session_state.username}
              res = requests.post(
                "http://localhost:8000/upload",
                files=files,
                data=data
              )
              if res.status_code == 200:
                st.success("✅ Document uploaded and indexed")
              else:
                st.error(f"❌ Upload failed: {res.json().get('error', 'Unknown error')}")
            except Exception as e:
              st.error(f"An error occurred: {str(e)}")

      with tab2:
        uploaded_files = st.file_uploader(
          "Upload multiple documents",
          type=["pdf", "docx", "csv"],
          accept_multiple_files=True
        )
        if uploaded_files and st.button("Process All Files"):
          with st.spinner("Processing..."):
            try:
              files = [("files[]", (file.name, file.getvalue())) for file in uploaded_files]
              data = {"username": st.session_state.username}
              res = requests.post(
                "http://localhost:8000/bulk-upload",
                files=files,
                data=data
              )
              if res.status_code == 200:
                result = res.json()
                st.success(f"✅ Processed {len(result.get('success', []))} files")
                if result.get("failed"):
                  st.warning(f"⚠ Failed: {', '.join(result['failed'])}")
              else:
                st.error(f"❌ Upload failed: {res.json().get('error', 'Unknown error')}")
            except Exception as e:
              st.error(f"An error occurred: {str(e)}")

  # Chat interface
  st.markdown("### Chat with the Policy Bot")

  # Chat history display
  with st.container():
    for i, (q, a, refs) in enumerate(st.session_state.conversation_history):
      # User question
      st.markdown(
        f'<div class="user-message">'
        f'<span class="message-number">Q{i + 1}:</span> {q}'
        f'</div>',
        unsafe_allow_html=True
      )
      # Bot answer
      st.markdown(
        f'<div class="bot-message">'
        f'<span class="message-number">A{i + 1}:</span> {a}'
        f'</div>',
        unsafe_allow_html=True
      )
      if refs:
       display_references(refs, i)

  # Chat input at bottom
  chat_input = st.chat_input("Type your question here...")


  if chat_input:
      with st.spinner("Thinking..."):
        try:
          res = requests.post(
            "http://localhost:8000/query",
            json={
              "question": chat_input,
              "session_id": st.session_state.session_id,
              "username": st.session_state.username,
              "language": st.session_state.language
            }
          )
          if res.status_code == 200:
            data = res.json()
            answer = data["answer"]
            references = data.get("references", [])
            st.session_state.conversation_history.append((chat_input, answer, references))
            st.session_state.faq_suggestions = data.get("suggestions", [])
            st.rerun() # Refresh to show new messages
          else:
            st.error(f"Error: {res.json().get('error', 'Unknown error')}")
        except Exception as e:
          st.error(f"Error: {str(e)}")

    # FAQ Suggestions - Moved outside the form submission block
  if st.session_state.get("faq_suggestions"):
      st.markdown("### 🔍 You might also ask:")
      for i, suggestion in enumerate(st.session_state.faq_suggestions[:4]): # Limit to 4 suggestions
        if st.button(suggestion, key=f"suggested_faq_{i}"):
          with st.spinner("Thinking..."):
            try:
              res = requests.post(
                "http://localhost:8000/query",
                json={
                    "question": suggestion,
                    "session_id": st.session_state.session_id,
                    "username": st.session_state.username,
                    "language": st.session_state.language
                  }
                )
              if res.status_code == 200:
                  data = res.json()
                  references = data.get("references", [])
                  st.session_state.conversation_history.append((suggestion, data["answer"], references))
                  st.session_state.faq_suggestions = data.get("suggestions", [])
                  st.rerun()

            except Exception as e:
              st.error(f"Error: {str(e)}")


