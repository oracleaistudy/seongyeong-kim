fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7,
31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5,
34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0,
38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 10.5, 10.6, 11.0, 11.2,
11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0,
450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0,
700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0,
925.0, 975.0, 950.0, 6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 
9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l,w] for l, w in zip (fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

# print (fish_data[4])
# print(fish_data[0:4])    #0부터 첫번째 글자로 해석

from sklearn.neighbors import KNeighborsClassifier        # KNneighborsClassifier 불러오기
kn = KNeighborsClassifier()              #해당 클래스에 대한 객체 생성

# train_input = fish_data[:35]        #처음부터 34개 
# train_target = fish_target[:35]
# test_input = fish_data[35:]                   #35개부터 끝 (14개)
# test_target = fish_target[35:]

# kn.fit(train_input, train_target)
# kn.score(test_input, test_target)

import numpy as np  

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

np.random.seed(42)          # 이 이후의 무작위작업(shuffle)이 동일한 결과를 내도록 기준지정
index = np.arange(49)    #인덱스 생성

np.random.shuffle(index)     #생성한 인덱스 무작위 섞음

print(input_arr[[1,3]])

train_input = input_arr[index[:35]]    #무작위 인덱스를 사용하여 34번째 값까지 등록
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]     
test_target = target_arr[index[35:]]

#print(test_target)
#print(test_input)
#print(train_input.shape)

import matplotlib.pyplot as plt

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
# plt.show()

kn = kn.fit(train_input, train_target)
kn.score(train_input, train_target)
kn.score(test_input, test_target)

# kn.predict([[25, 550]])
# kn.predict([[25, 150]])

kn.predict(test_input)

#kn.n_neighbors = 18
#kn = kn.fit(train_input, train_target)
#kn.score(train_input, train_target)
#kn.score(test_input, test_target) #훈련세트의 정확도보다 테스트 세트 정확도가 높을 수 있음
#kn.predict(test_input)


# for n in range(5, 35):
    kn.n_neighbors = n
    score = kn.score(train_input, train_target)
    if score <1:
        print(n, score)
        break
