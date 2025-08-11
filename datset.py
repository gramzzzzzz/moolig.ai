import csv

file_path = "/home/sem6/Downloads/AyurRagBot-main/dataset.csv"
file_path2 = "/home/sem6/Downloads/AyurRagBot-main/dataset_final.csv"

new_entry = [
    ("What is the Ayurvedic approach to hair health?", "Bhringraj, Amla, and Brahmi are commonly used herbs for promoting hair growth and preventing hair fall."),
    ("Which Ayurvedic text focuses on surgical procedures?", "The Sushruta Samhita is considered the foundational text of Ayurvedic surgery."),
    ("What is the role of Tulsi in Ayurveda?", "Tulsi (Holy Basil) is used for respiratory health, immunity, and stress relief."),
    ("How does Ayurveda describe joint disorders?", "Joint disorders are linked to Vata imbalance and treated with oil massages and herbal remedies like Guggulu."),
    ("What is the function of Shatavari in Ayurveda?", "Shatavari is an adaptogenic herb used for female reproductive health and hormone balance."),
    ("What is Pitta pacifying diet?", "A Pitta pacifying diet includes cooling foods like cucumber, coconut, and sweet fruits."),
    ("How does Ayurveda view hypertension?", "Hypertension is associated with aggravated Vata and Pitta and managed through diet, lifestyle, and herbs like Arjuna."),
    ("What is the function of Licorice in Ayurveda?", "Licorice (Yashtimadhu) is used for soothing the throat, digestion, and adrenal support."),
    ("How does Ayurveda classify pain?", "Pain (Shoola) is classified based on doshic involvement: Vata (sharp), Pitta (burning), and Kapha (dull)."),
    ("What is the Ayurvedic treatment for acidity?", "Herbs like Amla, Licorice, and cooling diets are recommended for acidity and hyperacidity."),
    ("What is the concept of Viruddha Ahara?", "Viruddha Ahara refers to incompatible food combinations that can cause diseases."),
    ("How does Ayurveda view anemia?", "Anemia (Pandu Roga) is considered a Pitta disorder and is treated with iron-rich foods and herbal supplements like Lohasava."),
    ("What is the Ayurvedic management of obesity?", "Obesity (Sthoulya) is managed by Kapha-reducing diets, exercise, and herbs like Guggulu and Triphala."),
    ("What are the benefits of Gokshura?", "Gokshura is used for kidney health, urinary disorders, and reproductive health."),
    ("How does Ayurveda explain fever?", "Fever (Jwara) is classified based on dosha involvement and treated with herbs like Guduchi and Sudarshana Churna."),
    ("What is the role of Sandalwood in Ayurveda?", "Sandalwood is used for cooling the body, skin health, and mental clarity."),
    ("How does Ayurveda treat menstrual disorders?", "Menstrual disorders are managed through diet, lifestyle, and herbs like Ashoka and Shatavari."),
    ("What is the function of Aloe Vera in Ayurveda?", "Aloe Vera is used for skin health, digestion, and wound healing."),
    ("How does Ayurveda approach cholesterol management?", "Cholesterol issues are linked to Kapha imbalance and treated with herbs like Guggulu and Arjuna."),
    ("What is the Ayurvedic view on mental clarity?", "Herbs like Brahmi, Gotu Kola, and Shankhapushpi enhance mental clarity and cognitive function."),
    ("What is the role of Ginger in Ayurveda?", "Ginger is widely used to improve digestion, circulation, and immune function."),
    ("What is the Ayurvedic perspective on asthma?", "Asthma is a Kapha-Vata disorder and is treated with Sitopaladi Churna and respiratory therapies."),
    ("What is the significance of Ashoka tree in Ayurveda?", "Ashoka tree bark is used for female reproductive health and menstrual regulation."),
    ("How does Ayurveda address stress management?", "Stress is managed using adaptogenic herbs like Ashwagandha, meditation, and yoga."),
    ("What is the Ayurvedic approach to liver disorders?", "Liver disorders are treated with herbs like Kutki, Bhumyamalaki, and Guduchi."),
    ("What are the benefits of Moringa in Ayurveda?", "Moringa is a superfood rich in nutrients, used for immunity and detoxification."),
    ("What is the concept of Dhatus in Ayurveda?", "Dhatus are the seven bodily tissues: Rasa, Rakta, Mamsa, Meda, Asthi, Majja, and Shukra."),
    ("How does Ayurveda classify headaches?", "Headaches are classified based on dosha involvement: Vata (migraines), Pitta (burning pain), and Kapha (dull pain)."),
    ("What is the Ayurvedic treatment for insomnia?", "Herbs like Brahmi, Jatamansi, and Ashwagandha are used for treating insomnia and sleep disorders."),
    ("How does Ayurveda define Ama Pachana?", "Ama Pachana refers to the process of digesting and eliminating toxins from the body using herbs and fasting."),
    ("What is the role of Neem in Ayurveda?", "Neem is used for blood purification, skin health, and immune support."),
    ("What are the benefits of Triphala Guggulu?", "Triphala Guggulu is used for weight management, detoxification, and digestion."),
    ("How does Ayurveda view arthritis?", "Arthritis is a Vata disorder and treated with herbal oils, massages, and anti-inflammatory herbs."),
    ("What is the function of Cinnamon in Ayurveda?", "Cinnamon is used to balance blood sugar, improve digestion, and enhance circulation."),
    ("What is the Ayurvedic view on fasting?", "Fasting is used to cleanse the body, balance doshas, and improve digestion."),
    ("How does Ayurveda treat sinusitis?", "Sinusitis is treated with Nasya therapy, steam inhalation, and herbs like Sitopaladi Churna."),
    ("What is the Ayurvedic approach to skin glow?", "A diet rich in antioxidants, hydration, and herbs like Turmeric and Saffron promote skin glow."),
    ("What is the significance of Bhallataka in Ayurveda?", "Bhallataka (Marking Nut) is used for respiratory disorders and immune enhancement."),
    ("What is the function of Cardamom in Ayurveda?", "Cardamom is a digestive stimulant and is used to balance Kapha and improve metabolism."),
    ("What is the Ayurvedic treatment for constipation?", "Triphala and castor oil are commonly used remedies for constipation."),
    ("What are the benefits of Black Pepper in Ayurveda?", "Black Pepper improves digestion, enhances bioavailability, and reduces Kapha congestion."),
    ("What is the role of Clove in Ayurveda?", "Clove is used for dental health, digestion, and respiratory issues."),
]

data = []
with open(file_path, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    data = list(reader)

data.extend(new_entry)

with open(file_path2, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(data)