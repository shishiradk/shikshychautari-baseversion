# Sidebar control
iters = st.sidebar.number_input("Reinforcement iterations (N)", min_value=5, max_value=200, value=100, step=5)

# On click
with st.spinner(f"Generating best-of-{iters} paper..."):
    past_db = load_vector_store()
    paper, score = generate_best_of_n(past_db, st.session_state.syllabus_text, n_samples=int(iters))
    st.success(f"Best paper selected. Score={score:.3f}")
    with st.expander("📄 Predicted Paper (Best-of-N)"):
        render_question_paper(paper)
        if REPORTLAB_OK:
            pdf_bytes = paper_to_pdf_bytes("Predicted Question Paper", paper)
            st.download_button("⬇️ Download Paper as PDF", pdf_bytes, "predicted_paper.pdf", "application/pdf", use_container_width=True)
