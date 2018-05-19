import json

filename = open("training", "r+")
#ids = open("user_id.txt", "w+")
#app = open("app.txt", "w+")
l1 = open("l1.txt", "w+")
user_o = open("user_o.txt", "w+")
#l2 = open("l2.txt", "w+")

article_cats = ['1000288', '1000780', '1000031', '1001091', '1000637', '1000001', '1000661', '1000395', '6000007', '1000721', '1000931', '1000111', '1000992', '1000560', '1000267', '1001039', '1000123', '1000742', '1000374', '1000298', '1000044', '1000620', '8000001']
# article = []

# print(list_of_categories)
#
# for line in filename:
# 	temp_dict = json.loads(line.rstrip())
# 	if ("article_l1_categories" in temp_dict):
# 		categories = temp_dict["article_l1_categories"]
# 		flatten = [x["name"] for x in categories]
# 		for x in flatten:
# 			if x not in article:
# 				article.append(x)
# 		# for x in categories:
# 		# 	if x not in list_of_categories:
# 		# 		list_of_categories.append(x)
# print(article)
#print([x for x in sorted(list_of_categories)])
user = ""
vector = [1] * 23
for line in filename:
	temp_dict = json.loads(line.rstrip())
	if ("user_gp_frequency" in temp_dict):
		if (temp_dict["user_id"] != user): #If we get a new user
			if ("article_l1_categories" in temp_dict):
				categories = temp_dict["article_l1_categories"]
				print(vector)
				normalized = [x / sum(vector) for x in vector]
				write_string = " ".join([str(y) for y in normalized])
				#print("The vector is ", write_string)
				user_o.write(user + "\n")
				l1.write(write_string + "\n")
				user = temp_dict["user_id"]
				vector = [0] * 23
				print("New user\n")
				print("Category is ", categories)
				for x in range(len(categories)):
					idx = article_cats.index(temp_dict["article_l1_categories"][x]["name"])
					vector[idx] += temp_dict["article_l1_categories"][x]["weight"]
		else: #If the user is same
			if ("article_l1_categories" in temp_dict):
				categories = temp_dict["article_l1_categories"]
				print("Category is ", categories)
				for x in range(len(categories)):
					idx = article_cats.index(temp_dict["article_l1_categories"][x]["name"])
					vector[idx] += temp_dict["article_l1_categories"][x]["weight"]
					print(vector)

normalized = [x / sum(vector) for x in vector]
write_string = " ".join([str(y) for y in normalized])
print("\n\nFinal vector ", write_string)
print("\n\nFinal user ", user)
user_o.close()
l1.close()
filename.close()
