data = {'entities': {'PERSON': ['Barack Obama', 'Bill Clinton', 'Fox & Friends', 'George W. Bush', 'Jonathan Swan', 'Ogan', 'President Trump', 'Rob Porter', 'Sarah Huggabee'], 'ORGANIZATION': ['FBI', 'FOX NEWS channel', 'Fox & Friends', 'Fox News', 'TPM TV', 'White House', 'YouTube'], 'DATE': ['0000:00:00', '00:00:00', '02 12', '1518450860', '2018', 'February 12', 'June 15', 'Sunday', 'Tuesday', 'by Wednesday'], 'EVENT': ['Allegations of domestic abuse', 'Fox & Friends segment', 'Interview', 'Security clearance revoked']}}
entities = data["entities"]
list_ners = []
for key, value in entities.items():
    list_ners.extend(value)

print(list_ners)