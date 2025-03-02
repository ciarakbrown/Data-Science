import cleaner
import loader

# Put training set path here
training_setA = loader.loader("")
df = cleaner.kalman_fill(training_setA[0])
print(df.head)
