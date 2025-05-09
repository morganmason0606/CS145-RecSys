from recommender_analysis_visualization import MyRecommender
from pyspark.sql import SparkSession


def get_recommender():
    return TEMPLATE_MODEL()
    #REPLACE ME
    raise NotImplementedError() # return model here


SUBMISSION_METADATA = {
    "team_name": '") drop table uers; --',
    "members": (
        "Jonathan Tam 8318",
        "Sonia Teo 6908",
        "Morgan Mason 7359",
        "Cara Burgess 5435",
    ),
    "description": "TEMPLATE. REPLACE ME", 
}

class TEMPLATE_MODEL(MyRecommender):
    # keep inheritance, setting up spark, etc
    def __init__(self, seed=None, spark=None):
        super().__init__(seed=seed)
        if spark is None:
            self.spark = (
                SparkSession.builder.appName("RecSysCompetition")
                .master("local[*]")
                .config("spark.driver.memory", "4g")
                .config("spark.sql.shuffle.partitions", "8")
                .getOrCreate()
            )
        else: 
            self.spark = spark 
        

    def fit(self, log, user_features=None, item_features=None):
        print(log)
        super().fit(log, user_features=user_features, item_features=item_features)

    def predict(
        self,
        log,
        k,
        users,
        items,
        user_features=None,
        item_features=None,
        filter_seen_items=True,
    ):

        return super().predict(
            log,
            k=k,
            users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items,
        )
