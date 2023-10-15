def get_train_data():

    dataset_train = []
    
    f = open('TableBank_data/Detection/annotations/tablebank_latex_train.json',)
    data = json.load(f)
    data["categories"][0]["id"] = 0

    count = 0

    for image_dict in data["images"][:10000]:
        
        #if count%1000 == 0:
        #    print(count)

        image = {}
        
        image_name = "TableBank_data/Detection/images/" + image_dict["file_name"]

        image["file_name"] = image_name
        image["height"] = image_dict["height"]
        image["width"] = image_dict["width"]
        image["image_id"] = image_dict["id"]
        image["id"] = image_dict["id"]

        annotations = []
        categories = []

        matches = list(filter(lambda person: person['image_id'] == image_dict["id"], data["annotations"]))
        
        for i in range(len(matches)):
            matches[i]["bbox_mode"] = BoxMode.XYWH_ABS
            matches[i]["category_id"] = 0
        
        image["annotations"] = matches
        image["categories"] = data["categories"]
        
        dataset_train.append(image)        
        
        #count += 1

    return dataset_train


def get_valid_data():

    dataset_train = []
    
    f = open('TableBank_data/Detection/annotations/tablebank_latex_val.json',)
    data = json.load(f)
    data["categories"][0]["id"] = 0

    count = 0

    for image_dict in data["images"][:5000]:
        
        #if count%1000 == 0:
        #    print(count)

        image = {}
        
        image_name = "TableBank_data/Detection/images/" + image_dict["file_name"]

        image["file_name"] = image_name
        image["height"] = image_dict["height"]
        image["width"] = image_dict["width"]
        image["image_id"] = image_dict["id"]
        image["id"] = image_dict["id"]

        annotations = []
        categories = []

        matches = list(filter(lambda person: person['image_id'] == image_dict["id"], data["annotations"]))
        
        for i in range(len(matches)):
            matches[i]["bbox_mode"] = BoxMode.XYWH_ABS
            matches[i]["category_id"] = 0
        
        image["annotations"] = matches
        image["categories"] = data["categories"]
        
        dataset_train.append(image)        
        
        #count += 1

    return dataset_train