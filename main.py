from octis.models.ETM import ETM
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

dataset = Dataset()
dataset.fetch_dataset("20NewsGroup")

model = ETM(num_topics=20)

output = model.train_model(dataset)
print(*list(output.keys()), sep="\n") # Print the output identifiers

for t in output['topics'][:5]:
    print(" ".join(t))


# Initialize metric
npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')

# Initialize metric
topic_diversity = TopicDiversity(topk=10)

# Retrieve metrics score
topic_diversity_score = topic_diversity.score(output)
print("Topic diversity: " + str(topic_diversity_score))

npmi_score = npmi.score(output)
print("Coherence: " + str(npmi_score))