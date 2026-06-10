window.DEMO_DATA = {
 "order": [
  "micro"
 ],
 "models": {
  "micro": {
   "label": "bge-micro-v2",
   "hf": "TaylorAI/bge-micro-v2",
   "params": "17.4M",
   "best_epoch": 2,
   "metrics": {
    "baseline": {
     "recall@1": 0.5138477526630294,
     "recall@5": 0.7939334892179787,
     "recall@10": 0.8666017147310989,
     "ndcg@10": 0.7754916374708859
    },
    "finetuned": {
     "recall@1": 0.5382826708235906,
     "recall@5": 0.8383086515978175,
     "recall@10": 0.9125227331774487,
     "ndcg@10": 0.8170145554221354
    }
   }
  }
 },
 "examples": {
  "micro": [
   {
    "query": "what are some of the hormonal effects of resistance exercise on the body?",
    "style": "informal",
    "baseline": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0284",
     "score": 0.8541134595870972,
     "gold_rank": 35,
     "rank1_ok": false
    },
    "finetuned": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0380",
     "score": 0.49928295612335205,
     "gold_rank": 1,
     "rank1_ok": true
    }
   },
   {
    "query": "how does being sick, having a cast, or getting injured affect muscle mass and strength?",
    "style": "informal",
    "baseline": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_1656",
     "score": 0.7171998023986816,
     "gold_rank": 20,
     "rank1_ok": false
    },
    "finetuned": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0416",
     "score": 0.34113219380378723,
     "gold_rank": 1,
     "rank1_ok": true
    }
   },
   {
    "query": "why is ammonia a problem for athletes?",
    "style": "informal",
    "baseline": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_1211",
     "score": 0.7123575210571289,
     "gold_rank": 10,
     "rank1_ok": false
    },
    "finetuned": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0215",
     "score": 0.4955194592475891,
     "gold_rank": 1,
     "rank1_ok": true
    }
   },
   {
    "query": "what hormones regulate fluid balance and kidney function?",
    "style": "informal",
    "baseline": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0340",
     "score": 0.735565185546875,
     "gold_rank": 9,
     "rank1_ok": false
    },
    "finetuned": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0301",
     "score": 0.516796350479126,
     "gold_rank": 1,
     "rank1_ok": true
    }
   },
   {
    "query": "what are some bad sprinting habits that can come from doing a lot of sprint conditioning?",
    "style": "informal",
    "baseline": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0770",
     "score": 0.7659286260604858,
     "gold_rank": 17,
     "rank1_ok": false
    },
    "finetuned": {
     "chunk_id": "NCSA_Essentials_of_ Strength_Training_and_Conditioning_2672",
     "score": 0.4491514265537262,
     "gold_rank": 1,
     "rank1_ok": true
    }
   }
  ]
 },
 "chunks": {
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0340": {
   "content": "Fluid volume shifts tend to occur from the blood to the intercellular compartment and the cells as a result of exercise. This shift can increase hormone concentrations in the blood without any change in secretion from endocrine glands. It has been hypothesized that, regardless of the mechanism of increase, such concentration changes increase receptor interaction probabilities.",
   "summary": "Exercise-induced fluid shifts can temporarily elevate blood hormone concentrations, potentially influencing receptor interactions."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0284": {
   "content": "It is important for strength and conditioning professionals to have a basic understanding of the hormonal responses to resistance exercise. Knowledge of hormonal responses to resistance exercise increases insight into how an exercise prescription can enable hormones to mediate optimal adaptations to resistance training. Although resistance training is the only natural stimulus that causes increases in lean tissue mass, dramatic differences exist among resistance training programs in their ability to produce increases in muscle and connective tissue size. The type of resistance training workout used dictates the hormonal responses. Tissue adaptations are influenced by the changes in circulating hormonal concentrations following exercise. Understanding this natural anabolic activity that takes place in the athlete’s body is fundamental to successful recovery, adaptation, program design, training progression, and ultimately athletic performance. It has been theorized that the endocrine system can be manipulated naturally with resistance training to enhance the development of various target tissues, thereby improving performance.",
   "summary": "This chunk emphasizes the importance of understanding hormonal responses to resistance exercise for optimizing training adaptations and athletic performance."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0416": {
   "content": "In situations of disease, joint immobilization, or injury, an elevation in cortisol mediates a nitrogen-wasting effect with a net loss of contractile protein. This results in muscle atrophy, with associated reductions in force production capability (38, 112).",
   "summary": "Elevated cortisol levels during disease, immobilization, or injury contribute to muscle atrophy and reduced force production."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0380": {
   "content": "Endocrine Responses to Resistance Exercise increases lipolysis (fat breakdown). Endocrine Responses to Resistance Exercise increases the availability of glucose and amino acids. Endocrine Responses to Resistance Exercise increases collagen synthesis. Endocrine Responses to Resistance Exercise stimulates cartilage growth. Endocrine Responses to Resistance Exercise increases retention of nitrogen, sodium, potassium, and phosphorus. Endocrine Responses to Resistance Exercise increases renal plasma flow and filtration. Endocrine Responses to Resistance Exercise promotes compensatory renal hypertrophy. Endocrine Responses to Resistance Exercise enhances immune cell function.",
   "summary": "Resistance exercise triggers a variety of hormonal responses that promote muscle growth, tissue repair, and overall physiological adaptation."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0770": {
   "content": "Simultaneous sprint and aerobic endurance training decreased sprint speed and jump power.  The physiological mechanisms involved in such responses to simultaneous training remain unclear. Such adaptations may result in overtraining. Other potential factors may include inadequate recovery between workouts and residual fatigue from aerobic workouts during resistance exercise.",
   "summary": "Simultaneous sprint and aerobic endurance training negatively impacts performance and may lead to overtraining due to unclear physiological mechanisms and potential recovery issues."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_1211": {
   "content": "Some athletes may not effectively manage their physical resources because of perceived incompetence or lack of self-efficacy. This is particularly unfortunate when physical tests and a coach’s judgment indicate superior potential.",
   "summary": "This chunk addresses the issue of athletes underperforming due to perceived limitations."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0215": {
   "content": "The nitrogenous waste products of amino acid degradation are eliminated through the formation of urea and small amounts of ammonia. The elimination through formation of ammonia is significant because ammonia is toxic and is associated with fatigue.",
   "summary": "This chunk explains the nitrogenous waste products of amino acid metabolism and their potential impact on fatigue."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_0301": {
   "content": "Atrial peptide, secreted by the heart (atrium), regulates sodium, potassium, and fluid volume. Renin, secreted by the kidney, regulates kidney function and permeability.",
   "summary": "The heart and kidneys secrete hormones that regulate fluid balance and kidney function."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_1656": {
   "content": "Coaches and trainers are equipped to deal with physical injuries and illnesses but are not trained to deal with mental illness, nor should they be. It is not the responsibility of the strength and conditioning professional to treat or diagnose an eating disorder. It is his or her ethical responsibility to assist the athlete in attaining diagnosis and treatment. When an athlete is suspected of having an eating disorder, strength and conditioning professionals should know their responsibilities and know when referral is appropriate.",
   "summary": "Strength and conditioning professionals should not treat eating disorders but should assist athletes in seeking proper diagnosis and treatment."
  },
  "NCSA_Essentials_of_ Strength_Training_and_Conditioning_2672": {
   "content": "Trunk/thigh weakness; fatigue\nHead and neck hyperextended or hyperflexed\nNormal erect head carriage, eyes focused ahead\nSome athletes' prior speed training experience consists of fatiguing sprint conditioning, which can reinforce unsound mechanics.",
   "summary": "These describe common sprinting technique issues related to fatigue, posture, and prior training."
  }
 }
};
