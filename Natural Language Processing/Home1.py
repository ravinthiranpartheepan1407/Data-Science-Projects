import datefinder

Detect_dates_by_Ravinthiran = """ I am having Machine Learning Lecture on 30/September/2019 and Examination dated on 25/10/2019 """

detection = datefinder.find_dates(Detect_dates_by_Ravinthiran)

for date_detection in detection :
    print (date_detection)
