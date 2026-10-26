import io
import html
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

st.set_page_config(
    page_title="Molecule Virtual Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

MOLECULES = {
    "Paracetamol": {
        "smiles": "CC(=O)NC1=CC=C(C=C1)O",
        "formula_note": "Commonly used as a pain reliever and fever reducer.",
        "benefits": [
            "Used for temporary relief of mild to moderate pain.",
            "Used to reduce fever.",
            "Does not have the same anti-inflammatory effect as aspirin.",
        ],
        "caution": "Taking too much paracetamol can cause severe liver damage. Follow the product label and advice from a qualified healthcare professional.",
    },
    "Aspirin": {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "formula_note": "An analgesic, antipyretic and anti-inflammatory medicine; it also affects platelet aggregation.",
        "benefits": [
            "Can relieve pain and reduce fever.",
            "Can reduce inflammation.",
            "In specific clinical situations, low-dose aspirin may be prescribed to reduce clotting risk.",
        ],
        "caution": "Aspirin can cause stomach irritation and bleeding, and is not suitable for everyone. Children and teenagers should not use it for viral illnesses unless specifically directed by a clinician.",
    },
    "Caffeine": {
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "formula_note": "A central nervous system stimulant found naturally in coffee, tea and some other products.",
        "benefits": [
            "Can temporarily increase alertness and reduce feelings of tiredness.",
            "Can improve attention for some people in suitable amounts.",
            "Has research uses in studying stimulant effects and molecular structure-property relationships.",
        ],
        "caution": "Too much caffeine can cause anxiety, sleep problems, palpitations or other unwanted effects. Sensitivity varies, and adolescents should be particularly cautious.",
    },
}

# ---------- Helpers ----------
def parse_smiles(smiles):
    """Return an RDKit molecule, or None when the SMILES cannot be parsed."""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return mol

def property_dict(mol):
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molar mass (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact mass (Da)": round(Descriptors.ExactMolWt(mol), 4),
        "Heavy atoms": int(mol.GetNumHeavyAtoms()),
        "H-bond donors": int(Lipinski.NumHDonors(mol)),
        "H-bond acceptors": int(Lipinski.NumHAcceptors(mol)),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "TPSA (Å²)": round(Descriptors.TPSA(mol), 3),
        "LogP (calculated)": round(Descriptors.MolLogP(mol), 3),
        "Ring count": int(rdMolDescriptors.CalcNumRings(mol)),
    }

def mol_to_sdf(mol):
    """Create a 3D conformer and return SDF text for the browser viewer."""
    mol3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    result = AllChem.EmbedMolecule(mol3d, params)
    if result != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=200)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol3d)

def render_3d(mol, style_name="Ball and stick", show_hydrogens=True):
    sdf = mol_to_sdf(mol)
    if not sdf:
        st.warning("A 3D conformer could not be generated for this structure.")
        return
    style_map = {
        "Ball and stick": "{stick:{radius:0.16},sphere:{scale:0.28}}",
        "Stick": "{stick:{radius:0.20}}",
        "Space filling": "{sphere:{scale:0.42}}",
        "Wireframe": "{line:{linewidth:2}}",
    }
    style = style_map.get(style_name, style_map["Ball and stick"])
    sdf_js = sdf.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    hydrogen_js = "true" if show_hydrogens else "false"
    page = f"""
    <div id="viewer" style="width:100%;height:440px;position:relative;background:#0b1220;border-radius:14px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      const element = document.getElementById('viewer');
      const viewer = $3Dmol.createViewer(element, {{backgroundColor:'#0b1220'}});
      const sdf = `{sdf_js}`;
      viewer.addModel(sdf, 'sdf');
      const model = viewer.getModel();
      if (!{hydrogen_js}) {{
        model.setStyle({{elem:'H'}}, {{stick:{{radius:0.001}}, sphere:{{scale:0.001}}}});
      }}
      model.setStyle({{}}, {style});
      viewer.zoomTo();
      viewer.render();
      viewer.zoom(0.9);
    </script>
    """
    components.html(page, height=455, scrolling=False)

def structure_image(mol, size=(650, 400)):
    clean = Chem.Mol(mol)
    try:
        Chem.rdDepictor.Compute2DCoords(clean)
    except Exception:
        pass
    return Draw.MolToImage(clean, size=size, kekulize=True)

def report_text(student_name, reg_no, selected_names, custom_smiles=None):
    lines = [
        "MOLECULE VIRTUAL LAB — FINAL PROJECT REPORT",
        "=" * 48,
        f"Student name: {student_name or 'Not provided'}",
        f"Register number: {reg_no or 'Not provided'}",
        "",
        "Purpose:",
        "To explore molecular structures, validate SMILES notation, inspect selected molecular descriptors,",
        "compare molecules, and review educational information about their common uses.",
        "",
        "Important note:",
        "Calculated descriptors are computational estimates based on molecular structure. This application",
        "is an educational tool and does not provide medical advice or treatment recommendations.",
        "",
        "Molecules studied:",
    ]
    for name in selected_names:
        smiles = MOLECULES[name]["smiles"]
        mol = parse_smiles(smiles)
        lines.append(f"\n{name}")
        lines.append(f"SMILES: {smiles}")
        for key, value in property_dict(mol).items():
            lines.append(f"{key}: {value}")
        lines.append("Educational uses / context:")
        lines.extend([f"- {x}" for x in MOLECULES[name]["benefits"]])
        lines.append(f"Caution: {MOLECULES[name]['caution']}")
    if custom_smiles:
        lines.extend(["", "Custom SMILES entered:", custom_smiles])
        custom_mol = parse_smiles(custom_smiles)
        if custom_mol:
            lines.extend([f"{k}: {v}" for k, v in property_dict(custom_mol).items()])
    lines.extend(["", "Generated by Molecule Virtual Lab."])
    return "\n".join(lines)

# ---------- Download helpers ----------
def _image_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def _excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Molecular properties")
    return buffer.getvalue()

def _pdf_bytes(text):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import cm
        from xml.sax.saxutils import escape
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.7*cm, leftMargin=1.7*cm, topMargin=1.7*cm, bottomMargin=1.7*cm)
        styles = getSampleStyleSheet()
        story = []
        for line in text.splitlines():
            if line.strip() == "=" * 48:
                story.append(Spacer(1, 8))
            elif line.strip():
                story.append(Paragraph(escape(line), styles["BodyText"]))
                story.append(Spacer(1, 4))
            else:
                story.append(Spacer(1, 5))
        doc.build(story)
        return buffer.getvalue()
    except Exception:
        return b"PDF generation unavailable. Please download the TXT report instead."


# ---------- Sidebar ----------
st.sidebar.title("🧪 Molecule Virtual Lab")
st.sidebar.caption("Explore • Visualize • Compare • Learn")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📘 Theory",
        "🧬 Molecular Visualization",
        "📊 Molecular Properties Report",
        "📈 Structure–Property Analysis",
        "📝 Assessment",
        "📄 Final Project Report",
    ],
)
st.sidebar.divider()
st.sidebar.subheader("👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student name", key="student_name", placeholder="Enter your name")
register_number = st.sidebar.text_input("Register number", key="register_number", placeholder="Enter your register number")
st.sidebar.caption("Student details stay in this app session and are included only in reports you download. Do not enter details on a shared/public device.")
st.sidebar.divider()
st.sidebar.caption("Educational chemistry tool • Not medical advice")

# ---------- Pages ----------
if page == "🏠 Home":
    st.title("🧪 Molecule Virtual Lab")
    st.write("An interactive chemistry learning app for exploring three familiar molecules: **paracetamol, aspirin and caffeine**.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Molecules included", "3")
    with c2:
        st.metric("Core learning tools", "6+")
    with c3:
        st.metric("Structure input", "SMILES")
    st.subheader("What can students do?")
    features = [
        ("Enter SMILES notation", "Paste a SMILES string to inspect another valid molecular structure."),
        ("Validate structures", "Check whether the entered SMILES can be parsed by RDKit."),
        ("Generate 2D structures", "View a clear structural diagram."),
        ("Explore interactive 3D structures", "Rotate and zoom molecular models in the browser."),
        ("Change visualization styles", "Choose ball-and-stick, stick, space-filling or wireframe display."),
        ("Calculate molecular properties", "View formula, molar mass, exact mass, hydrogen-bond counts, TPSA and more."),
        ("Download reports", "Export property data and a student-labelled project report."),
        ("Compare molecules", "Compare the three included molecules side by side."),
        ("Complete an assessment", "Test your understanding with a short quiz."),
    ]
    for title, desc in features:
        st.markdown(f"- **{title}:** {desc}")
    st.subheader("Included molecules")
    cols = st.columns(3)
    for col, name in zip(cols, MOLECULES):
        mol = parse_smiles(MOLECULES[name]["smiles"])
        with col:
            st.markdown(f"**{name}**")
            st.image(structure_image(mol, (400, 250)), use_container_width=True)
            st.caption(f"SMILES: `{MOLECULES[name]['smiles']}`")
    st.info("Use the navigation menu on the left to open the viewer, inspect properties, compare structures, or take the assessment.")

elif page == "📘 Theory":
    st.title("📘 Theory: Understanding Molecules")
    st.markdown("""
    **What is a molecule?**  
    A molecule is a group of atoms connected by chemical bonds. Its structure influences its physical and chemical properties.

    **What is SMILES?**  
    SMILES (Simplified Molecular Input Line Entry System) is a text notation used to represent molecular structures. For example, a SMILES string can encode atoms, bonds, branches and rings.

    **2D versus 3D structures**
    - A **2D structure** shows atoms and bonds on a flat page.
    - A **3D model** gives a spatial view of a generated molecular conformation. It is a model, not a guarantee of the molecule's exact shape in every real-world environment.

    **Structure–property relationships**  
    Features such as molecular size, polarity, hydrogen-bond donors and acceptors, and flexibility can help explain some molecular behaviour. No single descriptor fully predicts a molecule's biological effects.

    **RDKit descriptors**  
    The app uses RDKit to parse SMILES and calculate common structure-based descriptors. Values such as LogP and TPSA are computed estimates, not experimental measurements.
    """)
    st.subheader("Molecules in this lab")
    for name, data in MOLECULES.items():
        with st.expander(name):
            st.write(data["formula_note"])
            st.write("**SMILES:**", data["smiles"])
            st.write("**Educational context / common uses:**")
            for benefit in data["benefits"]:
                st.markdown(f"- {benefit}")
            st.warning(data["caution"])

elif page == "🧬 Molecular Visualization":
    st.title("🧬 Molecular Visualization")
    choice = st.selectbox("Choose a molecule", list(MOLECULES.keys()))
    custom_mode = st.checkbox("Use my own SMILES instead")
    if custom_mode:
        custom_smiles = st.text_input("Enter SMILES notation", placeholder="Paste a SMILES string here")
        mol = parse_smiles(custom_smiles)
        if mol is None:
            st.info("Enter a valid SMILES string to generate its structure.")
        else:
            st.success("SMILES validated successfully.")
            display_name = "Custom molecule"
    else:
        custom_smiles = ""
        mol = parse_smiles(MOLECULES[choice]["smiles"])
        display_name = choice
    if mol is not None:
        left, right = st.columns([1, 1.15])
        with left:
            st.subheader("2D structure")
            st.image(structure_image(mol), use_container_width=True)
            st.caption(f"SMILES: `{Chem.MolToSmiles(mol)}`")
        with right:
            st.subheader("Interactive 3D structure")
            style_name = st.selectbox("Visualization style", ["Ball and stick", "Stick", "Space filling", "Wireframe"])
            show_h = st.checkbox("Show hydrogen atoms", value=True)
            render_3d(mol, style_name, show_h)
            st.caption("Use your mouse or touchpad to rotate and zoom. The 3D conformer is computationally generated.")
        with st.expander("Quick properties"):
            st.dataframe(pd.DataFrame([property_dict(mol)]).T.rename(columns={0: "Value"}), use_container_width=True)
        st.download_button(
            "⬇️ Download 2D structure (PNG)",
            data=structure_image(mol).tobytes() if False else _image_bytes(structure_image(mol)),
            file_name=f"{display_name.lower().replace(' ', '_')}_structure.png",
            mime="image/png",
        )

elif page == "📊 Molecular Properties Report":
    st.title("📊 Molecular Properties Report")
    st.write("Select one or more molecules to inspect their calculated descriptors.")
    selected = st.multiselect("Molecules", list(MOLECULES.keys()), default=list(MOLECULES.keys()))
    if selected:
        rows = []
        for name in selected:
            mol = parse_smiles(MOLECULES[name]["smiles"])
            row = {"Molecule": name, "SMILES": MOLECULES[name]["smiles"]}
            row.update(property_dict(mol))
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download properties as CSV", df.to_csv(index=False).encode("utf-8"), "molecular_properties.csv", "text/csv")
        st.download_button("⬇️ Download properties as Excel", _excel_bytes(df), "molecular_properties.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.subheader("Educational uses and cautions")
        for name in selected:
            with st.expander(name):
                st.write(MOLECULES[name]["formula_note"])
                st.markdown("**Potential benefits / common uses**")
                for item in MOLECULES[name]["benefits"]:
                    st.markdown(f"- {item}")
                st.warning(MOLECULES[name]["caution"])
        st.caption("Descriptors are computed from the encoded structure. They should not be interpreted as proof of safety, effectiveness, or a clinical recommendation.")

elif page == "📈 Structure–Property Analysis":
    st.title("📈 Structure–Property Analysis")
    st.write("Compare computed molecular descriptors for the three included molecules.")
    rows = []
    for name, data in MOLECULES.items():
        mol = parse_smiles(data["smiles"])
        row = {"Molecule": name}
        row.update(property_dict(mol))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Molecule")
    chosen_property = st.selectbox(
        "Choose a descriptor to chart",
        ["Molar mass (g/mol)", "Exact mass (Da)", "Heavy atoms", "H-bond donors", "H-bond acceptors", "Rotatable bonds", "TPSA (Å²)", "LogP (calculated)", "Ring count"],
    )
    st.bar_chart(df[chosen_property])
    st.subheader("Side-by-side comparison")
    st.dataframe(df.T, use_container_width=True)
    st.markdown("""
    **How to interpret this carefully**
    - Molecular mass reflects the total atomic mass in a molecule.
    - Hydrogen-bond donor and acceptor counts describe certain structural features relevant to intermolecular interactions.
    - TPSA is a calculated estimate of polar surface area.
    - Calculated LogP estimates lipophilicity; it is not a direct measurement.
    - These descriptors alone cannot determine a medicine's effectiveness, dosage or safety.
    """)
    st.download_button("⬇️ Download comparison CSV", df.reset_index().to_csv(index=False).encode("utf-8"), "structure_property_comparison.csv", "text/csv")

elif page == "📝 Assessment":
    st.title("📝 Molecular Science Assessment")
    st.write("Answer the questions and submit to see your score and explanations.")
    questions = [
        {
            "q": "1. What does SMILES represent?",
            "options": ["A text notation for molecular structures", "A unit of molar mass", "A type of chemical bond", "A 3D printer format only"],
            "answer": "A text notation for molecular structures",
            "why": "SMILES encodes atoms and their connectivity in a text string.",
        },
        {
            "q": "2. Which molecule is commonly used as a stimulant in coffee and tea?",
            "options": ["Aspirin", "Caffeine", "Paracetamol", "All of them equally"],
            "answer": "Caffeine",
            "why": "Caffeine is a central nervous system stimulant.",
        },
        {
            "q": "3. Which tool is used in this app to parse SMILES and calculate descriptors?",
            "options": ["RDKit", "A spreadsheet alone", "A web browser alone", "A word processor"],
            "answer": "RDKit",
            "why": "RDKit is a cheminformatics toolkit for molecular structures and descriptors.",
        },
        {
            "q": "4. What does a 3D conformer in this app show?",
            "options": ["A computationally generated spatial model", "The exact shape in every environment", "A medical dosage", "The molecule's colour in real life"],
            "answer": "A computationally generated spatial model",
            "why": "Generated conformers are useful visual models, but do not represent every real-world condition.",
        },
        {
            "q": "5. Which statement about calculated descriptors is correct?",
            "options": ["They prove a drug is safe", "They replace clinical advice", "They describe selected structural features but do not prove clinical effects", "They always match experimental values exactly"],
            "answer": "They describe selected structural features but do not prove clinical effects",
            "why": "Descriptors help compare structures but cannot establish clinical safety or effectiveness on their own.",
        },
        {
            "q": "6. Which medicine has anti-inflammatory effects and also affects platelet aggregation?",
            "options": ["Aspirin", "Caffeine", "Water", "None of these"],
            "answer": "Aspirin",
            "why": "Aspirin has anti-inflammatory effects and inhibits platelet function; its use depends on clinical context.",
        },
    ]
    with st.form("assessment_form"):
        answers = {}
        for i, item in enumerate(questions):
            answers[i] = st.radio(item["q"], item["options"], index=None, key=f"q_{i}")
        submitted = st.form_submit_button("Submit assessment")
    if submitted:
        score = sum(answers[i] == item["answer"] for i, item in enumerate(questions))
        st.subheader(f"Your score: {score} / {len(questions)}")
        st.progress(score / len(questions))
        for i, item in enumerate(questions):
            if answers[i] == item["answer"]:
                st.success(f"{item['q']} — Correct. {item['why']}")
            else:
                st.error(f"{item['q']} — Correct answer: **{item['answer']}**. {item['why']}")
        st.caption("This is a learning quiz, not a formal or proctored assessment.")

elif page == "📄 Final Project Report":
    st.title("📄 Final Project Report")
    st.write("Generate a report that includes the student details entered in the sidebar and the three molecules.")
    st.text_input("Student name (from sidebar)", value=student_name, disabled=True)
    st.text_input("Register number (from sidebar)", value=register_number, disabled=True)
    selected_for_report = st.multiselect("Include molecules", list(MOLECULES.keys()), default=list(MOLECULES.keys()))
    include_custom = st.checkbox("Include a custom SMILES in the report")
    custom_for_report = ""
    if include_custom:
        custom_for_report = st.text_input("Custom SMILES", placeholder="Paste a valid SMILES string")
        if custom_for_report and parse_smiles(custom_for_report) is None:
            st.error("That SMILES could not be validated. Correct it before downloading.")
    report = report_text(student_name.strip(), register_number.strip(), selected_for_report, custom_for_report if include_custom and parse_smiles(custom_for_report) else None)
    st.text_area("Report preview", report, height=360)
    st.download_button("⬇️ Download final report (.txt)", report.encode("utf-8"), "molecule_virtual_lab_report.txt", "text/plain")
    st.download_button("⬇️ Download final report (.pdf)", _pdf_bytes(report), "molecule_virtual_lab_report.pdf", "application/pdf")
    st.info("Only include student details if your instructor requires them. Reports are generated when you click download; this app does not intentionally save them to a database.")

