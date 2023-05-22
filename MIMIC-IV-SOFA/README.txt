1. pn_patients.ipynb，根据入排标准筛选人群，单纯感染的每次住院仅入住1次ICU的
    a.分组：细菌性肺炎和病毒性肺炎
    b.包含出入院、入住ICU、spsis、死亡事件的发生和时间
2. describtion.ipynb:
    a.入院后到发生出院、入住ICU、sepsis、死亡事件等结局事件的生存曲线
3. prediction.ipynb:
    a.转化feature：
       a:人口学特征
       b:生命体征
       c:既往史
       d:辅助检查
            # {"gas": ['Specimen Type','Temperature',
            #         'pH','Base Excess','Anion Gap',
            #         'pO2','Oxygen','Oxygen Saturation',
            #         'pCO2','Calculated Total CO2','Bicarbonate',
            #         'Free Calcium','Potassium, Whole Blood','Glucose','Lactate',
            #         'Tidal Volume','Intubated','PEEP','Ventilation Rate','Ventilator'],
            # "blood":['Red Blood Cells', 'Hematocrit','Hemoglobin','RDW-SD', 'RDW',
            #         'MCH','MCHC','MCV',
            #         'White Blood Cells',
            #         'Monocytes','Neutrophils','Basophils','Eosinophils','Lymphocytes',
            #         'Absolute Lymphocyte Count','Absolute Monocyte Count','Absolute Neutrophil Count','Absolute Basophil Count','Absolute Eosinophil Count',
            #         'Immature Granulocytes','Atypical Lymphocytes','Metamyelocytes','Myelocytes',
            #         'Bands','Platelet Count','PT','PTT','INR(PT)'],
            # "axin":['Calcium, Total','Free Calcium','Phosphate','Magnesium','Potassium','Sodium','Chloride'],
            # "liver":['Alanine Aminotransferase (ALT)','Asparate Aminotransferase (AST)', 'Alkaline Phosphatase', 'Albumin', 'Bilirubin, Total','Bilirubin, Indirect', 'Glucose','Creatine Kinase (CK)','Creatine Kinase, MB Isoenzyme','Troponin T', 'Lipase','Lactate Dehydrogenase (LD)','Vancomycin'],
            # "kidney":['Creatinine','Urea Nitrogen']}
       e:病原种类
       f:影像学
       g:sofa评分

    
