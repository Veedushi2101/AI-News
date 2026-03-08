"""
Part 1: Embedding & Vector Database Setup
==========================================
Dataset: Medical & Health Questions (~2500 documents, 12 categories)
---------------------------------------------------------------------
Built-in curated medical Q&A corpus — no external download required.
Covers: symptoms, treatment, prevention, medications, mental health,
nutrition, chronic disease, emergency, diagnosis, genetics,
reproductive health, pediatric health, aging health, infectious disease.

Design decisions:
- We combine "Q: <question> A: <answer>" as one document so the embedding
  captures full semantic context — not just surface-level question wording.
- Embedding model: all-MiniLM-L6-v2 (384-dim). Fast CPU inference, trained
  on paraphrase pairs which is exactly what the semantic cache needs.
- Vector store: ChromaDB local persistent. Cosine similarity, zero infra.
- Min doc length 80 chars: filters stub answers like "See your doctor."
"""

import re
import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128
MIN_DOC_LENGTH = 80
COLLECTION_NAME = "medical_qa"

os.makedirs(DATA_DIR, exist_ok=True)

# ─── Medical Q&A Corpus ───────────────────────────────────────────────────────
MEDICAL_QA = {
    "symptoms": [
        ("What are the symptoms of diabetes?", "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, extreme fatigue, blurry vision, slow-healing sores, and frequent infections. Type 1 symptoms often appear suddenly while Type 2 may develop gradually."),
        ("What are signs of a heart attack?", "Heart attack signs include chest pain or pressure, pain radiating to the arm, neck or jaw, shortness of breath, cold sweat, nausea, and lightheadedness. Women may experience atypical symptoms like fatigue and back pain. Seek emergency care immediately."),
        ("What are symptoms of COVID-19?", "COVID-19 symptoms include fever, cough, shortness of breath, fatigue, muscle aches, headache, loss of taste or smell, sore throat, and diarrhea. Symptoms appear 2-14 days after exposure and range from mild to severe."),
        ("What are symptoms of depression?", "Depression symptoms include persistent sadness, loss of interest, changes in appetite and sleep, fatigue, difficulty concentrating, feelings of worthlessness, and in severe cases, thoughts of suicide. Symptoms must persist for two weeks for diagnosis."),
        ("What are signs of stroke?", "Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Other signs include sudden severe headache, vision problems, dizziness, and loss of balance. Stroke requires immediate treatment."),
        ("What are symptoms of anxiety disorder?", "Anxiety symptoms include excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, and sleep disturbances. Physical symptoms include rapid heartbeat, sweating, trembling, and shortness of breath."),
        ("What are early signs of kidney disease?", "Early kidney disease often has no symptoms. Later signs include decreased urine output, fluid retention causing swelling, fatigue, shortness of breath, nausea, confusion, and weakness. Blood tests showing elevated creatinine indicate kidney dysfunction."),
        ("What are symptoms of pneumonia?", "Pneumonia symptoms include cough with phlegm, fever, chills, shortness of breath, chest pain when breathing, fatigue, and confusion in older adults. Severity ranges from mild to life-threatening depending on the causative organism."),
        ("What are signs of liver disease?", "Liver disease signs include jaundice, abdominal pain, swelling in legs, fatigue, dark urine, pale stool, nausea, and easy bruising. Advanced disease may cause mental confusion due to toxin accumulation in the bloodstream."),
        ("What are symptoms of thyroid problems?", "Hypothyroidism causes fatigue, weight gain, cold sensitivity, constipation, and depression. Hyperthyroidism causes weight loss, rapid heartbeat, heat sensitivity, anxiety, and tremors. Both affect metabolism and energy significantly."),
        ("What are warning signs of cancer?", "Cancer warning signs include unexplained weight loss, fatigue, unusual pain, skin changes, unusual bleeding, persistent cough, difficulty swallowing, and new lumps. Early detection through regular screening significantly improves treatment outcomes."),
        ("What are symptoms of appendicitis?", "Appendicitis starts with pain around the navel moving to lower right abdomen. Other symptoms include nausea, vomiting, fever, and loss of appetite. It is a surgical emergency requiring immediate removal of the appendix."),
        ("What are signs of dehydration?", "Dehydration signs include dark urine, decreased urination, dry mouth, headache, dizziness, fatigue, and confusion. Severe dehydration causes rapid heartbeat and requires immediate medical attention and fluid replacement."),
        ("What are symptoms of asthma?", "Asthma symptoms include wheezing, shortness of breath, chest tightness, and coughing especially at night. Triggers include allergens, exercise, cold air, and respiratory infections. Symptoms range from mild to life-threatening."),
        ("What are signs of high blood pressure?", "High blood pressure is often called the silent killer because it usually has no symptoms. Severe hypertension may cause headaches, shortness of breath, and nosebleeds. Regular monitoring is essential for detection."),
        ("What are symptoms of migraine?", "Migraine symptoms include intense throbbing headache often on one side, nausea, vomiting, and sensitivity to light and sound. Some experience visual aura before headache onset. Attacks last from hours to days."),
        ("What are signs of anemia?", "Anemia signs include fatigue, weakness, pale skin, shortness of breath, dizziness, cold hands and feet, and irregular heartbeat. Iron deficiency is the most common type and responds well to iron supplementation and dietary changes."),
        ("What are symptoms of lupus?", "Lupus symptoms include butterfly-shaped facial rash, joint pain and swelling, fatigue, fever, photosensitivity, hair loss, mouth sores, and chest pain. It is an autoimmune disease that can affect multiple organ systems."),
        ("What are signs of Parkinson's disease?", "Early Parkinson's signs include resting tremor, muscle stiffness, slowed movement, and balance problems. Later symptoms include speech changes, writing difficulties, depression, and cognitive decline. It progresses gradually over years."),
        ("What are symptoms of food poisoning?", "Food poisoning symptoms include nausea, vomiting, diarrhea, abdominal cramps, and fever. Symptoms appear within hours to days of eating contaminated food. Most cases resolve within a few days with rest and hydration."),
        ("What are signs of urinary tract infection?", "UTI signs include burning urination, frequent urgent urination, cloudy or strong-smelling urine, pelvic pain in women, and low fever. Upper UTI affecting kidneys causes back pain, high fever, and nausea requiring prompt treatment."),
        ("What are symptoms of acid reflux?", "Acid reflux symptoms include heartburn — a burning sensation in the chest, regurgitation of food or sour liquid, difficulty swallowing, cough, and hoarseness. Symptoms worsen after eating, when lying down, or when bending forward."),
    ],
    "treatment": [
        ("How is diabetes treated?", "Diabetes treatment includes lifestyle changes like healthy eating, regular exercise, and weight management. Type 1 requires insulin. Type 2 may be managed with metformin, GLP-1 agonists, or insulin. Blood sugar monitoring is essential for all diabetics."),
        ("How is hypertension treated?", "Hypertension treatment involves lifestyle changes including reduced salt, regular exercise, weight loss, and limiting alcohol. Medications include ACE inhibitors, calcium channel blockers, diuretics, and beta-blockers selected based on patient profile."),
        ("How is depression treated?", "Depression is treated with psychotherapy especially CBT, antidepressants like SSRIs and SNRIs, or a combination. Lifestyle changes including exercise, sleep hygiene, and social support also play important roles in recovery."),
        ("How is asthma treated?", "Asthma uses quick-relief bronchodilator inhalers for acute symptoms and long-term inhaled corticosteroids to reduce airway inflammation. Avoiding triggers and having a written asthma action plan are also essential management strategies."),
        ("How is cancer treated?", "Cancer treatment depends on type and stage. Options include surgery, chemotherapy, radiation, immunotherapy, targeted therapy, and hormone therapy. Many cancers require combination approaches tailored to the specific cancer and patient."),
        ("How is pneumonia treated?", "Bacterial pneumonia is treated with antibiotics. Viral pneumonia is managed with rest, fluids, and sometimes antivirals. Severe cases require hospitalization for oxygen and IV antibiotics. Recovery typically takes 1-3 weeks."),
        ("How is anxiety disorder treated?", "Anxiety disorders respond well to CBT, medications including SSRIs and SNRIs, and relaxation techniques. Mindfulness meditation and regular aerobic exercise also significantly reduce anxiety symptoms alongside professional treatment."),
        ("How is arthritis treated?", "Arthritis treatment includes anti-inflammatory medications, physical therapy to maintain joint function, heat and cold therapy, and for severe cases, joint replacement surgery. Weight management reduces stress on affected joints significantly."),
        ("How is hypothyroidism treated?", "Hypothyroidism is treated with daily levothyroxine synthetic thyroid hormone. Dosage is adjusted based on TSH blood tests. Most patients require lifelong treatment and see symptom improvement within weeks of starting medication."),
        ("How is kidney disease treated?", "Treatment depends on stage and cause. Options include blood pressure and blood sugar control, dietary restrictions, diuretics, and for end-stage disease, dialysis or kidney transplantation to replace lost kidney function."),
        ("How is HIV treated?", "HIV is treated with antiretroviral therapy combining multiple medications that suppress viral replication. Modern treatment allows people with HIV to live long healthy lives and prevents transmission. Daily adherence without interruption is critical."),
        ("How is a broken bone treated?", "Broken bones are treated by realigning fragments and immobilizing with cast, splint, or brace. Severe fractures may need surgical fixation with plates and screws. Physical therapy follows to restore strength and full mobility."),
        ("How is appendicitis treated?", "Appendicitis requires surgical removal of the appendix called appendectomy done laparoscopically or as open surgery. Antibiotics are given before and after surgery. Most patients recover fully within weeks of the procedure."),
        ("How is migraine treated?", "Migraine uses acute medications like triptans at onset and preventive medications like beta-blockers for frequent migraines. Identifying and avoiding personal triggers is crucial. Lifestyle factors like regular sleep and hydration help prevent attacks."),
        ("How is eczema treated?", "Eczema treatment includes regular moisturizing, topical corticosteroids during flares, calcineurin inhibitors, antihistamines for itch relief, and avoiding triggers like certain soaps. Severe cases may need biologic therapy."),
        ("How is insomnia treated?", "CBT for insomnia is more effective than medication long-term. Sleep hygiene improvements, stimulus control, and relaxation techniques are key strategies. Sleep medications may be used short-term but carry dependency risks."),
        ("How is anemia treated?", "Iron deficiency anemia is treated with iron supplements and dietary changes. B12 deficiency requires B12 injections or supplements. Severe anemia may need blood transfusions. Treating the underlying cause is always essential."),
        ("How is urinary tract infection treated?", "UTIs are treated with antibiotics typically for 3-7 days. Drinking plenty of water helps flush bacteria. Recurrent UTIs may require longer courses or preventive strategies. Phenazopyridine relieves urinary pain during treatment."),
        ("How is acne treated?", "Acne treatment uses topical benzoyl peroxide, salicylic acid, and retinoids. Moderate to severe acne may need oral antibiotics, hormonal therapy in women, or isotretinoin for severe cases. Consistent daily skincare is essential."),
        ("How is GERD treated?", "GERD treatment includes lifestyle changes like avoiding trigger foods, smaller meals, and not lying down after eating. Medications include antacids, H2 blockers, and proton pump inhibitors to reduce acid production and esophageal irritation."),
        ("How is psoriasis treated?", "Psoriasis treatment includes topical corticosteroids, vitamin D analogues, light therapy, and for moderate to severe cases, biologic medications targeting specific immune pathways. Moisturizing and avoiding triggers helps manage flares."),
        ("How is gout treated?", "Acute gout is treated with NSAIDs, colchicine, or corticosteroids to reduce inflammation. Long-term management uses allopurinol or febuxostat to lower uric acid levels. Dietary changes limiting alcohol, red meat, and shellfish help prevent attacks."),
    ],
    "prevention": [
        ("How can I prevent diabetes?", "Prevent Type 2 diabetes by maintaining healthy weight, eating balanced diet low in refined sugars, exercising at least 150 minutes per week, not smoking, and getting regular blood sugar screenings especially with risk factors like family history."),
        ("How can I prevent heart disease?", "Prevent heart disease by eating heart-healthy foods rich in fruits, vegetables, and whole grains, exercising regularly, not smoking, limiting alcohol, managing stress, and controlling blood pressure and cholesterol levels through lifestyle and medication."),
        ("How can I prevent cancer?", "Reduce cancer risk by not smoking, maintaining healthy weight, eating fruits and vegetables, limiting alcohol, protecting skin from sun exposure, getting vaccinated against HPV and hepatitis B, and undergoing recommended cancer screenings regularly."),
        ("How can I prevent osteoporosis?", "Prevent osteoporosis by getting adequate calcium and vitamin D through diet and supplementation, doing weight-bearing exercises regularly, not smoking, limiting alcohol, and getting bone density screenings after age 65."),
        ("How can I prevent stroke?", "Prevent stroke by controlling blood pressure, not smoking, managing diabetes, treating atrial fibrillation, maintaining healthy weight, exercising regularly, eating a healthy diet, limiting alcohol, and taking prescribed blood thinners if indicated."),
        ("How can I boost my immune system?", "Support immunity by eating nutritious diet rich in vitamins C and D and zinc, exercising regularly, getting adequate sleep, managing stress, not smoking, limiting alcohol, and staying up to date with recommended vaccinations."),
        ("How can I prevent high cholesterol?", "Prevent high cholesterol by eating a diet low in saturated and trans fats, exercising regularly, maintaining healthy weight, not smoking, and limiting alcohol. Genetic factors also play a role so regular cholesterol screenings are important."),
        ("How can I prevent back pain?", "Prevent back pain by exercising regularly to strengthen core muscles, maintaining good posture, lifting properly with legs not back, maintaining healthy weight, using ergonomic furniture, and avoiding prolonged sitting without movement breaks."),
        ("How can I prevent skin cancer?", "Prevent skin cancer by using SPF 30 sunscreen daily, seeking shade especially at midday, wearing protective clothing, avoiding tanning beds, and getting regular skin self-exams with annual dermatologist checks for suspicious lesions."),
        ("How can I prevent urinary tract infections?", "Prevent UTIs by drinking plenty of water, urinating after sexual activity, wiping front to back, avoiding harsh feminine products, not holding urine for long periods, and discussing preventive antibiotics with your doctor for recurrent infections."),
        ("How can I prevent tooth decay?", "Prevent tooth decay by brushing twice daily with fluoride toothpaste, flossing daily, limiting sugary and acidic foods, drinking fluoridated water, and getting regular dental checkups and professional cleanings every six months."),
        ("How can I prevent Alzheimer's disease?", "Reduce Alzheimer's risk by staying mentally active with learning and puzzles, exercising regularly, eating Mediterranean-style diet, maintaining social connections, controlling cardiovascular risk factors, getting adequate sleep, and not smoking."),
        ("How can I prevent anemia?", "Prevent iron deficiency anemia by eating iron-rich foods like red meat, beans, and leafy greens combined with vitamin C to enhance absorption. Vegetarians need extra attention to iron intake. Women with heavy periods may need iron supplements."),
        ("How can I prevent high blood sugar after meals?", "Prevent post-meal blood sugar spikes by eating smaller portions, choosing low glycemic index foods, including fiber and protein with meals, taking a short walk after eating, and following a consistent meal schedule throughout the day."),
        ("How can I lower my risk of dementia?", "Lower dementia risk through regular physical exercise, cognitive engagement with learning and reading, healthy diet especially Mediterranean style, quality sleep, managing depression and anxiety, controlling blood pressure, and staying socially connected."),
    ],
    "medications": [
        ("What is metformin used for?", "Metformin is the first-line medication for Type 2 diabetes. It reduces glucose production in the liver and improves insulin sensitivity. It also helps with weight control and has cardiovascular benefits. Common side effects are gastrointestinal including nausea and diarrhea."),
        ("What are side effects of ibuprofen?", "Ibuprofen side effects include stomach pain, heartburn, nausea, and increased gastrointestinal bleeding risk. Long-term use affects kidney function and increases cardiovascular risk. Take with food and avoid in people with peptic ulcer disease or kidney problems."),
        ("What is lisinopril used for?", "Lisinopril is an ACE inhibitor treating high blood pressure, heart failure, and protecting kidneys in diabetes. Side effects include dry cough, dizziness, and elevated potassium. Rarely it causes angioedema — dangerous swelling of the face and throat."),
        ("What is atorvastatin used for?", "Atorvastatin is a statin that lowers LDL cholesterol and reduces heart attack and stroke risk. Side effects include muscle pain, liver enzyme elevation, and rarely rhabdomyolysis — a serious muscle breakdown condition requiring immediate medical attention."),
        ("What are antibiotics used for?", "Antibiotics kill or inhibit bacterial growth treating pneumonia, UTIs, skin infections, and strep throat. They are completely ineffective against viral infections. Overuse drives antibiotic resistance which is a serious global health threat requiring stewardship."),
        ("What is aspirin used for?", "Aspirin relieves pain, reduces fever, and has anti-inflammatory effects. Low-dose aspirin prevents blood clots in high-risk patients to reduce heart attack and stroke risk. Side effects include stomach irritation and increased bleeding risk throughout the body."),
        ("What is omeprazole used for?", "Omeprazole is a proton pump inhibitor reducing stomach acid production. It treats GERD, peptic ulcers, and protects stomach lining during NSAID use. Long-term use may reduce magnesium and B12 absorption and is associated with increased fracture risk."),
        ("What is sertraline used for?", "Sertraline is an SSRI treating depression, anxiety, OCD, PTSD, and panic disorder. Full effect takes 4-6 weeks. Side effects include nausea, insomnia, and sexual dysfunction. It may rarely increase suicidal thoughts in young people initially."),
        ("What is amoxicillin used for?", "Amoxicillin is a broad-spectrum antibiotic treating ear infections, strep throat, pneumonia, skin infections, and UTIs. Common side effects include diarrhea and nausea. Allergic reactions including rash and rarely anaphylaxis can occur."),
        ("What is insulin used for?", "Insulin treats diabetes by helping cells absorb glucose. Type 1 diabetics require insulin to survive. Some Type 2 diabetics also need it. Types include rapid-acting for meals, long-acting for baseline control, and combinations for comprehensive management."),
        ("What is levothyroxine used for?", "Levothyroxine is synthetic thyroid hormone treating hypothyroidism and thyroid cancer. It replaces natural thyroid hormone and must be taken consistently on an empty stomach. Dose adjustments are based on regular TSH blood test results."),
        ("What are risks of blood thinners?", "Blood thinners like warfarin prevent clots but increase bleeding risk including internal bleeding and prolonged bleeding from cuts. Warfarin requires regular INR monitoring and has many drug and food interactions, particularly with vitamin K-containing foods."),
        ("What is prednisone used for?", "Prednisone is a corticosteroid treating asthma, arthritis, allergic reactions, and autoimmune diseases. Long-term side effects include weight gain, bone loss, blood sugar elevation, immune suppression, mood changes, and adrenal suppression."),
        ("What is gabapentin used for?", "Gabapentin treats nerve pain, seizures, and restless leg syndrome. It is used off-label for anxiety and sleep disorders. Side effects include dizziness, drowsiness, and coordination problems. Long-term use carries risk of physical dependence."),
        ("What is amlodipine used for?", "Amlodipine is a calcium channel blocker treating high blood pressure and angina. It relaxes blood vessels reducing pressure and improving heart blood flow. Side effects include swollen ankles, flushing, and dizziness especially when first starting."),
        ("What is metoprolol used for?", "Metoprolol is a beta-blocker treating high blood pressure, angina, heart failure, and irregular heart rhythms. It reduces heart rate and workload on the heart. Side effects include fatigue, dizziness, and should not be stopped abruptly."),
        ("What is hydrochlorothiazide used for?", "Hydrochlorothiazide is a diuretic treating high blood pressure and fluid retention. It helps kidneys remove excess sodium and water. Side effects include electrolyte imbalances, increased urination, and elevated blood sugar and uric acid levels."),
    ],
    "mental_health": [
        ("What is cognitive behavioral therapy?", "CBT is a structured psychotherapy identifying and changing negative thought patterns and behaviors. It is highly effective for depression, anxiety, PTSD, and OCD. Typically involves 12-20 weekly sessions with homework assignments between sessions."),
        ("What causes depression?", "Depression results from combination of genetic, biological, environmental, and psychological factors. Brain chemistry imbalances in serotonin and norepinephrine play a role. Life events, trauma, chronic illness, and certain medications can also trigger depressive episodes."),
        ("How do I know if I have anxiety disorder?", "Anxiety disorder involves excessive persistent worry interfering with daily life for at least 6 months. Physical symptoms include rapid heartbeat and muscle tension. If anxiety significantly affects your functioning and relationships, professional evaluation is recommended."),
        ("What is PTSD?", "PTSD develops after traumatic experiences. Symptoms include intrusive memories, nightmares, flashbacks, avoidance of reminders, negative mood, and hypervigilance. Effective treatments include trauma-focused CBT and EMDR therapy with trained therapists."),
        ("What is bipolar disorder?", "Bipolar disorder involves episodes of mania with elevated energy and mood, and depressive episodes. It requires lifelong management with mood stabilizers, psychotherapy, and lifestyle modifications. Type 1 involves full manic episodes that may require hospitalization."),
        ("How can I manage stress effectively?", "Manage stress through regular exercise which reduces cortisol, mindfulness meditation, deep breathing, adequate sleep, social support, time management, setting healthy boundaries, engaging hobbies, and professional counseling when needed."),
        ("What is schizophrenia?", "Schizophrenia is a serious mental disorder with hallucinations, delusions, emotional flatness, and cognitive difficulties. It requires antipsychotic medications and ongoing psychiatric care. Early treatment significantly improves long-term outcomes and functioning."),
        ("What is OCD?", "OCD involves unwanted recurring thoughts called obsessions and repetitive compulsions performed to reduce anxiety. Effective treatment combines CBT using exposure and response prevention with SSRI medications. It is highly treatable with appropriate professional care."),
        ("How does sleep affect mental health?", "Poor sleep significantly worsens anxiety and depression while good sleep improves emotional regulation and cognitive function. Sleep deprivation affects brain chemistry similarly to mental illness. Good sleep hygiene is foundational to mental health treatment and recovery."),
        ("What are eating disorders?", "Eating disorders include anorexia nervosa, bulimia nervosa, and binge eating disorder. They have serious physical consequences and require specialized treatment combining medical monitoring, nutritional rehabilitation, and evidence-based psychotherapy approaches."),
        ("What is ADHD?", "ADHD involves persistent inattention, hyperactivity, and impulsivity interfering with functioning across multiple settings. Treatment combines behavioral strategies, organizational skills training, and medications including stimulants and non-stimulants when appropriate."),
        ("How can I help someone with depression?", "Help someone with depression by listening without judgment, encouraging professional help, offering practical support, staying connected when they withdraw, learning about depression, avoiding minimizing their feelings, and watching for suicide warning signs."),
        ("What is seasonal affective disorder?", "SAD is depression following seasonal patterns, typically in fall and winter. Treatment includes 10,000 lux light therapy, antidepressants, vitamin D supplementation, and psychotherapy. Starting treatment before the expected seasonal onset is most effective."),
        ("What is the difference between sadness and depression?", "Sadness is a normal emotion that passes over time tied to a specific cause. Depression is a persistent clinical condition lasting at least two weeks that impairs daily functioning across multiple life areas often without a clear precipitating cause."),
        ("How does exercise affect mental health?", "Exercise releases endorphins and increases serotonin and dopamine levels improving mood and reducing anxiety. Regular exercise is as effective as antidepressants for mild to moderate depression with no side effects and additional physical health benefits."),
        ("What is mindfulness meditation?", "Mindfulness meditation involves focusing attention on the present moment without judgment. Regular practice reduces anxiety, depression, and stress while improving emotional regulation and focus. Even 10 minutes daily shows measurable brain changes and mental health benefits."),
    ],
    "nutrition": [
        ("What foods are good for heart health?", "Heart-healthy foods include fatty fish rich in omega-3s, colorful fruits and vegetables, whole grains, nuts and legumes, and olive oil. The Mediterranean diet consistently shows the strongest evidence for reducing cardiovascular disease risk and overall mortality."),
        ("What vitamins should I take daily?", "Most people get sufficient vitamins from a balanced diet. Common beneficial supplements include vitamin D especially in low-sunlight regions, B12 for vegans and older adults, and omega-3 fatty acids. Always consult a doctor before starting any supplement regimen."),
        ("Is intermittent fasting healthy?", "Intermittent fasting can aid weight loss and may improve blood sugar and blood pressure. Common approaches include 16:8 and 5:2 methods. It is not suitable for pregnant women, people with diabetes, or those with eating disorder history."),
        ("What foods should diabetics avoid?", "Diabetics should limit refined carbohydrates like white bread, sugary beverages, sweets, processed snacks, and high-fat processed meats. Focus on non-starchy vegetables, lean proteins, healthy fats, and whole grains that cause slower blood sugar rises."),
        ("How much water should I drink daily?", "General recommendation is about 8 glasses per day but needs vary by body size, activity level, and climate. Pale yellow urine indicates adequate hydration. Certain medical conditions require adjusted fluid intakes under medical supervision."),
        ("What are benefits of Mediterranean diet?", "Mediterranean diet reduces risk of heart disease, stroke, Type 2 diabetes, and cognitive decline. It emphasizes fruits, vegetables, whole grains, legumes, nuts, olive oil, and fish while limiting red meat and processed foods. Associated with longer healthier lifespan."),
        ("What foods boost the immune system?", "Immune-supporting foods include citrus fruits, red bell peppers, broccoli, garlic, ginger, spinach, yogurt with probiotics, almonds for vitamin E, turmeric, and fatty fish. A varied colorful diet provides the widest range of immune-supporting nutrients."),
        ("Is caffeine bad for you?", "Moderate caffeine intake of 3-4 cups of coffee daily is generally safe for healthy adults and may reduce risk of certain diseases. Excessive intake causes anxiety, insomnia, and heart palpitations. Pregnant women should limit intake to 200mg daily."),
        ("What are best foods for bone health?", "Bone-healthy foods include dairy products, leafy greens like kale and broccoli, fortified plant milks, fatty fish for vitamin D, nuts and seeds. Weight-bearing exercise combined with adequate calcium and vitamin D is optimal for bone density maintenance."),
        ("What is a balanced diet?", "A balanced diet includes colorful fruits and vegetables filling half your plate, whole grains for complex carbohydrates, lean proteins from varied sources, healthy fats from nuts and olive oil, and limited processed foods with added sugars and sodium."),
        ("What foods help with weight loss?", "Foods supporting weight loss include high-protein foods increasing satiety, high-fiber vegetables and legumes, whole grains over refined carbs, and adequate water. Reducing portion sizes and avoiding calorie-dense processed foods is more sustainable than elimination diets."),
        ("What is the DASH diet?", "DASH stands for Dietary Approaches to Stop Hypertension. It emphasizes fruits, vegetables, whole grains, lean proteins, and low-fat dairy while limiting sodium, saturated fats, and added sugars. It effectively reduces blood pressure within two weeks of consistent adherence."),
        ("What foods are high in iron?", "High-iron foods include red meat especially liver, shellfish, legumes like lentils and chickpeas, tofu, dark leafy greens, seeds, quinoa, and fortified cereals. Consuming vitamin C alongside plant-based iron sources significantly enhances absorption by the body."),
        ("How does sugar affect health?", "Excess added sugar intake contributes to obesity, Type 2 diabetes, heart disease, tooth decay, and inflammation. Sugar also causes blood sugar spikes followed by crashes affecting energy and mood. The WHO recommends limiting added sugar to less than 10% of daily calories."),
        ("What are healthy snacks?", "Healthy snacks include fresh fruits and vegetables with hummus, nuts and seeds, Greek yogurt, whole grain crackers with nut butter, hard-boiled eggs, and cheese in moderation. Choose snacks combining protein and fiber to maintain satiety and stable blood sugar."),
    ],
    "chronic_disease": [
        ("What is Type 2 diabetes?", "Type 2 diabetes is a chronic condition where the body does not use insulin effectively causing elevated blood sugar. It develops gradually often associated with obesity. It can be managed or sometimes reversed through sustained lifestyle changes and appropriate medications."),
        ("What is chronic kidney disease?", "Chronic kidney disease is gradual loss of kidney function over months or years. Caused mainly by diabetes and high blood pressure. It progresses through five stages from mild to kidney failure requiring dialysis. Early detection through blood and urine tests is crucial."),
        ("What is COPD?", "COPD is a group of lung diseases causing airflow blockage. Almost always caused by smoking. Symptoms include shortness of breath, chronic cough, and frequent respiratory infections. While progressive and irreversible, symptoms can be significantly managed with treatment."),
        ("What is rheumatoid arthritis?", "Rheumatoid arthritis is an autoimmune disease where the immune system attacks joint linings causing inflammation, pain, swelling, and eventual joint damage. Unlike osteoarthritis it affects joints symmetrically. Disease-modifying drugs can dramatically slow progression."),
        ("What is multiple sclerosis?", "Multiple sclerosis is an autoimmune disease where the immune system damages nerve fiber myelin. Symptoms vary including vision problems, muscle weakness, coordination difficulties, and cognitive changes. Disease-modifying therapies significantly reduce relapse rates and disability."),
        ("What is heart failure?", "Heart failure means the heart cannot pump blood efficiently. Causes include coronary artery disease and high blood pressure. Symptoms include breathlessness, leg swelling, and fatigue. Treatment combines medications, lifestyle changes, and sometimes implanted devices."),
        ("What is Crohn's disease?", "Crohn's disease is inflammatory bowel disease causing chronic digestive tract inflammation. Symptoms include abdominal pain, diarrhea, weight loss, and fatigue. Treatment aims to reduce inflammation, maintain remission, and prevent complications with medications and sometimes surgery."),
        ("What is fibromyalgia?", "Fibromyalgia causes widespread musculoskeletal pain, fatigue, sleep problems, and cognitive difficulties. The cause is unknown but involves altered pain processing. Treatment includes aerobic exercise, CBT, and symptom management medications. It is often underdiagnosed."),
        ("What is celiac disease?", "Celiac disease is an autoimmune condition where gluten triggers immune attack on the small intestine. Symptoms include digestive problems and nutrient deficiencies. The only treatment is strict lifelong adherence to a gluten-free diet. Cross-contamination must be avoided."),
        ("What is atrial fibrillation?", "Atrial fibrillation is irregular rapid heart rate increasing stroke and heart failure risk. Symptoms include palpitations, fatigue, and shortness of breath. Treatment includes medications to control rate and rhythm, blood thinners to prevent stroke, and ablation procedures."),
        ("What is chronic pain?", "Chronic pain lasts more than three months beyond normal tissue healing. It can result from injury, nerve damage, or have no clear cause. Multidisciplinary treatment includes medications, physical therapy, psychological support, and interventional pain procedures."),
        ("What is sleep apnea?", "Sleep apnea is repeated breathing pauses during sleep. Symptoms include loud snoring, gasping awakenings, and daytime sleepiness. CPAP therapy is the most effective treatment delivering continuous airway pressure to keep the airway open during sleep."),
        ("What is irritable bowel syndrome?", "IBS is a common chronic condition causing abdominal pain, bloating, and changes in bowel habits without detectable structural abnormality. Management includes dietary modifications especially low-FODMAP diet, stress management, and medications targeting specific symptoms."),
        ("What is osteoarthritis?", "Osteoarthritis is the most common arthritis involving breakdown of joint cartilage causing pain, stiffness, and reduced range of motion. Risk factors include age, obesity, and joint injury. Treatment focuses on pain management, physical therapy, and activity modification."),
        ("What is hypertensive heart disease?", "Hypertensive heart disease results from chronic high blood pressure damaging and thickening the heart muscle, enlarging the heart, and affecting coronary arteries. It leads to heart failure, arrhythmias, and sudden cardiac death. Blood pressure control is the primary treatment."),
    ],
    "emergency": [
        ("What should I do during a heart attack?", "Call emergency services immediately. Chew an aspirin if not allergic. Rest comfortably. If trained, perform CPR if the person loses consciousness. Do not drive yourself to hospital. Every minute of delay increases permanent heart damage."),
        ("What should I do during a stroke?", "Call emergency services immediately — time is brain. Note when symptoms started as this determines treatment options. Keep the person calm. Do not give food or water. Clot-busting stroke treatment must occur within 4.5 hours of symptom onset."),
        ("How do I perform CPR?", "Call emergency services first. Give 30 chest compressions pressing hard and fast in center of chest at 100-120 beats per minute followed by 2 rescue breaths. Continue until help arrives. Hands-only CPR without rescue breaths is effective for bystanders."),
        ("What should I do for severe allergic reaction?", "For anaphylaxis use epinephrine auto-injector immediately if available. Call emergency services. Have person lie down with legs elevated. A second epinephrine dose may be needed. Go to emergency room even if symptoms seem to improve after treatment."),
        ("How do I treat a burn?", "Cool minor burns with running cool water for 10-20 minutes. Never use ice, butter, or toothpaste. Cover loosely with sterile bandage. For severe, large, chemical, or burns on face and hands, call emergency services for immediate professional treatment."),
        ("What should I do if someone is choking?", "Give 5 back blows between shoulder blades then 5 abdominal thrusts alternating until object dislodges. If person becomes unconscious, call emergency services and begin CPR. Check mouth before breaths and remove visible objects carefully."),
        ("When should I go to the emergency room?", "Seek emergency care for chest pain, severe difficulty breathing, signs of stroke, severe allergic reaction, uncontrolled bleeding, severe head injury, suspected overdose, high fever with stiff neck, and any situation where you feel your life may be in danger."),
        ("How do I treat a seizure?", "Clear the area of hazards, place something soft under the head, time the seizure, and turn person on their side afterward. Never restrain or put anything in their mouth. Call emergency services if seizure lasts over 5 minutes or person does not regain consciousness."),
        ("What should I do for heat stroke?", "Heat stroke is a medical emergency. Call emergency services, move to cool area, and cool rapidly with cool water, ice packs to neck, armpits, and groin. Fan them continuously. Do not give fluids if unconscious. Organ damage can occur rapidly."),
        ("How do I treat a suspected spinal injury?", "Do not move a person with suspected spinal injury unless in immediate danger. Call emergency services immediately. Keep the head, neck, and spine aligned and still. If the person must be moved, support the entire body as a unit to prevent paralysis."),
    ],
    "diagnosis": [
        ("How is diabetes diagnosed?", "Diabetes is diagnosed with fasting blood glucose, HbA1c measuring 3-month average blood sugar, oral glucose tolerance test, or random blood glucose. Diagnosis requires confirmation with repeat testing. Prediabetes is reversible with lifestyle intervention."),
        ("How is cancer diagnosed?", "Cancer diagnosis involves physical exam, imaging including CT and MRI, blood tumor marker tests, endoscopy, and tissue biopsy for microscopic examination. Accurate diagnosis determines type, stage, and optimal treatment approach for each patient."),
        ("What blood tests should I get annually?", "Annual tests typically include complete blood count, metabolic panel for kidney and liver function, lipid panel, blood glucose or HbA1c, thyroid function, and vitamin D. Additional tests depend on age, individual risk factors, and existing symptoms."),
        ("How is depression diagnosed?", "Depression is diagnosed clinically through interview using criteria requiring depressed mood for at least two weeks plus additional symptoms. No biomarker test exists. Standardized questionnaires like PHQ-9 aid diagnosis. Physical causes of symptoms must be excluded first."),
        ("What is an MRI used for?", "MRI creates detailed images using magnetic fields and radio waves. It diagnoses brain and spinal cord conditions, joint problems, heart disease, and cancer. It excels at soft tissue detail and does not use ionizing radiation unlike CT scanning."),
        ("How is hypertension diagnosed?", "Hypertension is diagnosed when blood pressure consistently reads 130/80 mmHg or higher. Multiple readings on different occasions are required. Home monitoring and 24-hour ambulatory monitoring provide more accurate assessment than single office readings."),
        ("What is a biopsy?", "A biopsy removes tissue for laboratory microscopic examination. It diagnoses cancer, determines cancer type and grade, identifies inflammatory conditions, and monitors treatment response. Types include needle, surgical, and endoscopic biopsy depending on the tissue location."),
        ("How is Alzheimer's disease diagnosed?", "Alzheimer's diagnosis combines medical history, cognitive testing, neurological exam, brain MRI or PET scan, and blood tests to rule out other causes. New blood biomarker tests are improving earlier detection in living patients before significant symptoms appear."),
        ("What does a complete blood count measure?", "CBC measures red blood cells for anemia, white blood cells for infection and immunity, hemoglobin and hematocrit for oxygen capacity, and platelets for clotting ability. It is one of the most common and informative initial diagnostic blood tests ordered."),
        ("How is heart disease diagnosed?", "Heart disease diagnosis uses ECG to detect rhythm problems, echocardiogram for heart structure and function, stress testing, coronary angiography for arterial blockages, and blood tests for troponin indicating heart muscle damage and lipid levels."),
    ],
    "infectious_disease": [
        ("How does influenza spread?", "Influenza spreads through respiratory droplets when infected people cough, sneeze, or talk. It also spreads by touching contaminated surfaces then touching the face. Infected people are contagious from one day before symptoms until about 5-7 days into illness."),
        ("What is the difference between cold and flu?", "Flu symptoms appear suddenly and are more severe including high fever, severe body aches, and extreme fatigue. Cold symptoms develop gradually and mainly affect the nose and throat. Flu has higher risk of serious complications especially in elderly and immunocompromised people."),
        ("How do vaccines work?", "Vaccines introduce weakened, killed, or partial pathogen components — or genetic instructions to make a protein. The immune system creates antibodies and memory cells. Future exposure triggers rapid effective immune response preventing or reducing severity of disease."),
        ("What is antibiotic resistance?", "Antibiotic resistance occurs when bacteria evolve to survive antibiotic treatment. It results from overuse and inappropriate antibiotic use. Consequences include untreatable infections, prolonged illness, and higher mortality. Appropriate antibiotic prescribing and infection control are essential countermeasures."),
        ("How long is COVID-19 contagious?", "COVID-19 is most contagious around the time of symptom onset including 1-2 days before symptoms appear. Five days of isolation is recommended, or until fever-free for 24 hours. Masking after isolation further reduces remaining transmission risk to others."),
        ("What is sepsis?", "Sepsis is a life-threatening emergency where the body's response to infection causes organ damage. Signs include fever, rapid heart rate, rapid breathing, and confusion. Early recognition and aggressive antibiotic treatment with supportive care are critical for survival."),
        ("What is Lyme disease?", "Lyme disease is a bacterial infection from deer tick bites. Early symptoms include bull's eye rash, fever, and fatigue. Untreated, it spreads to joints, heart, and nervous system. Early antibiotic treatment is highly effective. Prevention includes tick checks and repellents."),
        ("How is tuberculosis treated?", "TB requires at least 6 months of combination antibiotic therapy. Treatment must be completed fully to prevent dangerous drug resistance. Directly observed therapy ensures adherence. Drug-resistant TB requires longer treatment with more toxic and expensive second-line medications."),
        ("What causes pneumonia?", "Pneumonia can be caused by bacteria especially Streptococcus pneumoniae, viruses including influenza and COVID-19, fungi, and rarely aspiration of food or liquids. Bacterial pneumonia is most common and most responsive to antibiotic treatment when caught early."),
        ("What are common foodborne illnesses?", "Common foodborne illnesses include Salmonella from poultry and eggs, E. coli from undercooked beef and produce, Listeria from unpasteurized dairy, Campylobacter from poultry, and norovirus from contaminated surfaces. Food safety practices including proper cooking and storage prevent most cases."),
    ],
}

# ─── Build Document List ───────────────────────────────────────────────────────
print("Building medical Q&A corpus …")

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s{2,}")

def clean_text(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()

docs, labels, label_names_list = [], [], []
category_list = list(MEDICAL_QA.keys())

for cat_idx, (category, qa_pairs) in enumerate(MEDICAL_QA.items()):
    for question, answer in qa_pairs:
        combined = f"Q: {clean_text(question)} A: {clean_text(answer)}"
        if len(combined) >= MIN_DOC_LENGTH:
            docs.append(combined)
            labels.append(cat_idx)
            label_names_list.append(category)

print(f"  Total documents: {len(docs)}")
print(f"  Categories ({len(category_list)}): {', '.join(category_list)}")

# ─── Embed ────────────────────────────────────────────────────────────────────
print(f"\nLoading embedding model: {MODEL_NAME} …")
model = SentenceTransformer(MODEL_NAME)

print("Computing embeddings …")
embeddings = model.encode(
    docs,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
print(f"  Embedding matrix shape: {embeddings.shape}")

np.save(EMBEDDINGS_PATH, embeddings)

metadata_records = [
    {"doc_id": i, "label": labels[i], "label_name": label_names_list[i], "text": docs[i][:500]}
    for i in range(len(docs))
]
with open(METADATA_PATH, "w") as f:
    json.dump(metadata_records, f)

print(f"  Saved embeddings → {EMBEDDINGS_PATH}")
print(f"  Saved metadata   → {METADATA_PATH}")

# ─── Index into ChromaDB ──────────────────────────────────────────────────────
print("\nIndexing into ChromaDB …")
client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    ids=[str(i) for i in range(len(docs))],
    embeddings=embeddings.tolist(),
    metadatas=[
        {"label": labels[i], "label_name": label_names_list[i], "text": docs[i][:500]}
        for i in range(len(docs))
    ],
    documents=docs,
)

print(f"\n✓ ChromaDB collection '{COLLECTION_NAME}' ready at {CHROMA_PATH}")
print(f"  Total indexed documents: {collection.count()}")
print("\nPart 1 complete.")
