from Model.requirement import Requirement
from Analysis.quality_analyser import correct_requirements, correct_single_requirement

import os

class MockLLM:
    def get_response(self, messages, stream=False):
        class Msg:
            content = "the system shall work correctly"
        class Choice:
            message = Msg()
        class Resp:
            choices = [Choice()]
        return Resp()

llm = MockLLM()
req = Requirement(name="REQ-1", content="If raining, the system behaves quickly", state="Draft", asil="QM", rationale="")

res = correct_single_requirement(0, req, llm, None)
print(res)
