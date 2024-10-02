from src.models.builders import EndToEndGenerationBuilder

cite_strings = [
    {
        "string": "- These prohibitions can be temporary bans on specific water uses, such as using a hosepipe [1], [2], [3], [4], [6], [7], and [9]. Fin.",
        "cites": [1, 2, 3, 4, 6, 7, 9],
        "assertion_count": 2,
    },
    {
        "string": "- Water undertakers have the power to issue prohibitions, but they must comply with certain procedures, including giving notice of the prohibition and its terms [1, 2, 3]. Fin.",
        "cites": [1, 2, 3],
        "assertion_count": 2,
    },
    {
        "string": "- This doesn't cite anything. It's just a sentence. Fin. ",
        "cites": [],
        "assertion_count": 1,
    },
    {"string": "Single citation [1]. Fin.", "cites": [1], "assertion_count": 2},
    {
        "string": "- Citation 1 [1,2]. Citation 2 [3]. Fin.",
        "cites": [1, 2, 3],
        "assertion_count": 3,
    },
    {
        "string": "- Citation 1 [1,2]. Citation 2 [3] [5]. Citation 3 [4], [5], and [6]. Fin.",
        "cites": [1, 2, 3, 4, 5, 6],
        "assertion_count": 4,
    },
]


def test_assertion_extraction():
    for cite_string in cite_strings:
        generation_builder = EndToEndGenerationBuilder()

        print(cite_string["string"])
        generation_builder.set_answer(cite_string["string"])
        print(generation_builder.assertions)
        assert len(generation_builder.assertions) == cite_string["assertion_count"]
