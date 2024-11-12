import pytest
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

alarm_model = BayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

def create_query() :

    cpd_burglary = TabularCPD(
        variable="Burglary", variable_card=2, values=[[0.999], [0.001]],
        state_names={"Burglary":['no','yes']},
    )
    cpd_earthquake = TabularCPD(
        variable="Earthquake", variable_card=2, values=[[0.998], [0.002]],
        state_names={"Earthquake":["no","yes"]},
    )
    cpd_alarm = TabularCPD(
        variable="Alarm",
        variable_card=2,
        values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
        evidence=["Burglary", "Earthquake"],
        evidence_card=[2, 2],
        state_names={"Burglary":['no','yes'], "Earthquake":['no','yes'], 'Alarm':['yes','no']},
    )
    cpd_johncalls = TabularCPD(
        variable="JohnCalls",
        variable_card=2,
        values=[[0.95, 0.1], [0.05, 0.9]],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={"Alarm":['yes','no'], "JohnCalls":['yes', 'no']},
    )
    cpd_marycalls = TabularCPD(
        variable="MaryCalls",
        variable_card=2,
        values=[[0.1, 0.7], [0.9, 0.3]],
        evidence=["Alarm"],
        evidence_card=[2],
    state_names={"Alarm":['yes','no'], "MaryCalls":['yes', 'no']},
    )

    # Associating the parameters with the model structure
    alarm_model.add_cpds(
        cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)

    alarm_infer = VariableElimination(alarm_model)

    queries = []

    #print(alarm_infer.query(variables=["JohnCalls"],evidence={"Earthquake":"yes"}))
    q_1 = alarm_infer.query(variables=["JohnCalls", "Earthquake"],evidence={"Burglary":"yes","MaryCalls":"yes"})
    queries.append(q_1)
    #print(q_1)

    #Query 2- the probability of Mary Calling given that John called
    q_2 = alarm_infer.query(variables=["MaryCalls"],evidence={"JohnCalls":"yes"})
    #print(q_2)
    queries.append(q_2)

    #Query 3 - The probability of both John and Mary calling given Alarm
    q_3 = alarm_infer.query(variables=["MaryCalls", "JohnCalls"],evidence={"Alarm":"yes"})
    #print(q_3)
    queries.append(q_3)

    #Query 4 - The probability of Alarm, given that Mary called.
    q_4 = alarm_infer.query(variables=["Alarm"], evidence={"MaryCalls":"yes"})
    #print(q_4)
    queries.append(q_4)

    return queries

if __name__ == '__main__' :
    q_list = create_query()
    i = 1
    for q in q_list :
        print("Query:", i)
        print(q)
        i += 1
