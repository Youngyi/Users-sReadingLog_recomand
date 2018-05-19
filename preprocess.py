import json

filename = open("training", "r+")
ids = open("user_id.txt", "w+")
app = open("app.txt", "w+")


app_categories = ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY', 'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FAMILY_ACTION', 'FINANCE', 'FOOD_AND_DRINK', 'GAME_ACTION', 'GAME_ADVENTURE', 'GAME_ARCADE', 'GAME_BOARD', 'GAME_CARD', 'GAME_CASINO', 'GAME_CASUAL', 'GAME_EDUCATIONAL', 'GAME_MUSIC', 'GAME_PUZZLE', 'GAME_RACING', 'GAME_ROLE_PLAYING', 'GAME_SIMULATION', 'GAME_SPORTS', 'GAME_STRATEGY', 'GAME_TRIVIA', 'GAME_WORD', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME', 'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'MAPS_AND_NAVIGATION', 'MEDIA_AND_VIDEO', 'MEDICAL', 'MUSIC_AND_AUDIO', 'NEWS_AND_MAGAZINES', 'PARENTING', 'PERSONALIZATION', 'PHOTOGRAPHY', 'PRODUCTIVITY', 'SHOPPING', 'SOCIAL', 'SPORTS', 'TOOLS', 'TRANSPORTATION', 'TRAVEL_AND_LOCAL', 'VIDEO_PLAYERS', 'WEATHER']

# print(list_of_categories)
#
# for line in filename:
# 	temp_dict = json.loads(line.rstrip())
# 	if ("user_gp_frequency" in temp_dict):
# 		categories = temp_dict["user_gp_frequency"].keys()
# 		for x in categories:
# 			if x not in list_of_categories:
# 				list_of_categories.append(x)
#
# print([x for x in sorted(list_of_categories)])
user = ""
for line in filename:
	temp_dict = json.loads(line.rstrip())
	if ((temp_dict["user_id"] != user) and ("article_l1_categories" in temp_dict)):
		user = temp_dict["user_id"]
		if ("user_gp_frequency" in temp_dict):
			categories = temp_dict["user_gp_frequency"].keys()
			#vals = list(temp_dict["user_gp_frequency"].values())
			vector = [0] * 52
			for x in categories:
				idx = app_categories.index(x)
				vector[idx] = temp_dict["user_gp_frequency"][x]
			normalized = [x / sum(vector) for x in vector]
			write_string = " ".join([str(y) for y in normalized])
			#print("The vector is ", write_string)
			ids.write(user + "\n")
			app.write(write_string + "\n")
filename.close()
ids.close()
app.close()
