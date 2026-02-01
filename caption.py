from transformers import pipeline

captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

capt1 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.1})[0]['generated_text']
capt2 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.4})[0]['generated_text']
capt3 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.6})[0]['generated_text']
capt4 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.8})[0]['generated_text']
capt5 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 1.0})[0]['generated_text']
capt6 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 1.2})[0]['generated_text']
capt7 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 1.4})[0]['generated_text']
capt8 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 1.6})[0]['generated_text']
capt9 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 1.8})[0]['generated_text']
capt10 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature":1.9})[0]['generated_text']

print("Caption 1", capt1)
print("Caption 2", capt2)
print("Caption 3", capt3)
print("Caption 4", capt4)
print("Caption 5", capt5)
print("Caption 6", capt6)
print("Caption 7", capt7)
print("Caption 8", capt8)
print("Caption 9", capt9)
print("Caption 10", capt10)

capt11 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 10})[0]["generated_text"]
capt12 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 20})[0]["generated_text"]
capt13 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 30})[0]["generated_text"]
capt14 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 40})[0]["generated_text"]
capt15 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 50})[0]["generated_text"]
capt16 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 60})[0]["generated_text"]
capt17 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 70})[0]["generated_text"]
capt18 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 80})[0]["generated_text"]
capt19 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 90})[0]["generated_text"]
capt20 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_k": 100})[0]["generated_text"]

print("Caption 11", capt11)
print("Caption 12", capt12)
print("Caption 13", capt13)
print("Caption 14", capt14)
print("Caption 15", capt15)
print("Caption 16", capt16)
print("Caption 17", capt17)
print("Caption 18", capt18)
print("Caption 19", capt19)
print("Caption 20", capt20)

capt21 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 0.2})[0]["generated_text"]
capt22 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 0.5})[0]["generated_text"]
capt23 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 0.6})[0]["generated_text"]
capt24 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 0.8})[0]["generated_text"]
capt25 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 1.0})[0]["generated_text"]
capt26 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 1.2})[0]["generated_text"]
capt27 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 1.4})[0]["generated_text"]
capt28 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 1.6})[0]["generated_text"]
capt29 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 1.8})[0]["generated_text"]
capt30 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "top_p": 2.0})[0]["generated_text"]

print("Caption 21", capt21)
print("Caption 22", capt22)
print("Caption 23", capt23)
print("Caption 24", capt24)
print("Caption 25", capt25)
print("Caption 26", capt26)
print("Caption 27", capt27)
print("Caption 28", capt28)
print("Caption 29", capt29)
print("Caption 30", capt30)

capt31 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.1})[0]["generated_text"]
capt32 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.2})[0]["generated_text"]
capt33 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.3})[0]["generated_text"]
capt34 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.4})[0]["generated_text"]
capt35 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.5})[0]["generated_text"]
capt36 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.6})[0]["generated_text"]
capt37 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.7})[0]["generated_text"]
capt38 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.8})[0]["generated_text"]
capt39 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 0.9})[0]["generated_text"]
capt40 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "typical_p": 1.0})[0]["generated_text"]

print("Caption 31", capt31)
print("Caption 32", capt32)
print("Caption 33", capt33)
print("Caption 34", capt34)
print("Caption 35", capt35)
print("Caption 36", capt36)
print("Caption 37", capt37)
print("Caption 38", capt38)
print("Caption 39", capt39)
print("Caption 40", capt40)

capt41 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.1, "top_k": 20})[0]['generated_text']
capt42 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.5, "top_k": 50})[0]['generated_text']
capt43 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.9, "top_k": 90})[0]['generated_text']
capt44 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.1, "top_p": 1.0})[0]['generated_text']
capt45 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.1, "top_p": 0.1})[0]['generated_text']
capt46 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.9, "top_p": 0.9})[0]['generated_text']
capt47 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 1.0, "typical_p": 0.1})[0]['generated_text']
capt48 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.1, "typical_p": 1.0})[0]['generated_text']
capt49 = captioner("testing.jpg", generate_kwargs={"do_sample": True, "temperature": 0.9, "typical_p": 0.9})[0]['generated_text']
capt50 = captioner("testing.jpg")[0]['generated_text']

print("Caption 41", capt41)
print("Caption 42", capt42)
print("Caption 43", capt43)
print("Caption 44", capt44)
print("Caption 45", capt45)
print("Caption 46", capt46)
print("Caption 47", capt47)
print("Caption 48", capt48)
print("Caption 49", capt49)
print("Caption 50", capt50)