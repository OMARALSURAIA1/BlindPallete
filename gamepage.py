import streamlit as st
import random

# Custom CSS for styling and centering
st.markdown("""
<style>
/* Center all content */
.main-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}
/* Plate image sizing */
.plate-img img {
  width: 360px !important;
  height: auto !important;
  border-radius: 8px;
}
/* Filled buttons */
.stButton>button {
  background-color: #1f77b4;
  color: white;
  border-radius: 8px;
  height: 40px;
  font-size: 1rem;
  width: 100%;
  margin: 1px;
            }
/* Nothing button in red */
.nothing-button .stButton>button {
  background-color: #d62728;
}
</style>
""", unsafe_allow_html=True)

# Manually define plates and correct answers
plates = [
    {"path": f"plates/{i}.png", "answer": "Nothing" if i in [0,1,2,4,9] else str(i)}
    for i in range(10)
]

# Reset test: shuffle and clear answers
def reset_test():
    random.shuffle(plates)
    st.session_state.current = 0
    for i in range(len(plates)):
        st.session_state[f"answer_{i}"] = None

# Initialize session state
total = len(plates)
if "current" not in st.session_state:
    reset_test()
for idx in range(total):
    if f"answer_{idx}" not in st.session_state:
        st.session_state[f"answer_{idx}"] = None

# Record answer and advance
def record_answer(ans):
    idx = st.session_state.current
    st.session_state[f"answer_{idx}"] = ans
    st.session_state.current += 1

# Header
st.title("Color Blind Test")
st.write(f"Click the number you see (or 'Nothing') to proceed. Test ends after {total} plates.")

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

cur = st.session_state.current
if cur < total:
    # Display plate image
    st.markdown('<div class="plate-img">', unsafe_allow_html=True)
    st.image(plates[cur]["path"], use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # Restore original numpad layout
    rows = [["1","2","3"], ["4","5","6"], ["7","8","9"]]
    for row in rows:
        cols = st.columns(3, gap="small")
        for i, num in enumerate(row):
            cols[i].button(num, key=f"btn_{cur}_{num}", on_click=record_answer, args=(num,))

    # Nothing button
    st.markdown('<div class="nothing-button">', unsafe_allow_html=True)
    st.button("Nothing", key=f"btn_{cur}_none", on_click=record_answer, args=("Nothing",))
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Show score
    score = sum(
        1 for i, plate in enumerate(plates)
        if st.session_state[f"answer_{i}"] == plate["answer"]
    )
    st.write(f"**Your total score:** {score} / {total}")
    pct = score / total
    if pct >= 0.8:
        st.success("You are likely not color blind.")
    elif pct >= 0.5:
        st.warning("You may have some difficulty distinguishing certain colors.")
    else:
        st.error("You might be color blind. Consider consulting an eye specialist.")

    if st.button("Retake Test"):
        reset_test()

st.markdown('</div>', unsafe_allow_html=True)

# To run:
# 1. Place 0.png–9.png into a "plates" folder next to this script
# 2. pip install streamlit
# 3. streamlit run color_blind_test_app.py