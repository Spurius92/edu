"""
This is a file for stepik course 'introduction to python'

"""
# file names to open and edit
dataset_file = ''
submission_file = ''

with open(dataset_file, 'r') as dataset:
    with open(submission_file, 'w') as submit:
        total = [0, 0, 0]
        length = 0
        for data in dataset:
            grade = [int(i) for i in data.strip().split(';')[1:]]
            # print(grade)
            res = sum(grade) / 3
            print(res, file=submit, end='\n')
            # print('average grade is:', res)
            # submit.write(str(res))
            moment_average = list(map(sum, zip(grade, total)))
            total = moment_average
            length += 1
            # print(total)
        # submit.write('{} {} {}'.format(*average))
        total = [round(tot/length, 9) for tot in total]
        print(*total, file=submit)

# # check whether first part is working
# with open('E:\submission.txt', 'r') as dataset:
#     sub = dataset.read()
#     print(sub)