import dowhy
import pandas as pd
import networkx as nx
from typing import List

class AirApportionmentDAG:
    """
    Defines the Causal DAG for Air Apportionment.
    """
    def __init__(self):
        # We define a causal graph where meteorological variables confound 
        # the relationship between proxy treatments and PM2.5.
        
        self.treatments = ['proxy_traffic', 'proxy_industry', 'proxy_dust', 'proxy_biomass']
        self.confounders = ['temp', 'humidity', 'wind_u', 'wind_v', 'pblh']
        self.outcome = 'pm25'
        
    def generate_gml(self) -> str:
        """
        Generate the graph in GML format for DoWhy.
        """
        graph = nx.DiGraph()
        
        # Add nodes
        for t in self.treatments:
            graph.add_node(t)
        for c in self.confounders:
            graph.add_node(c)
        graph.add_node(self.outcome)
        
        # Confounders affect both treatments and the outcome
        for c in self.confounders:
            graph.add_edge(c, self.outcome)
            for t in self.treatments:
                graph.add_edge(c, t)
                
        # Treatments affect the outcome
        for t in self.treatments:
            graph.add_edge(t, self.outcome)
            
        # Convert to GML string
        gml = "".join(nx.generate_gml(graph))
        return gml
    
    def create_model(self, df: pd.DataFrame, treatment: str) -> dowhy.CausalModel:
        """
        Create a DoWhy CausalModel for a specific treatment proxy.
        
        Args:
            df: DataFrame containing the data.
            treatment: The specific treatment proxy to evaluate.
            
        Returns:
            DoWhy CausalModel.
        """
        if treatment not in self.treatments:
            raise ValueError(f"Treatment {treatment} must be one of {self.treatments}")
            
        # We can either use the GML graph, or specify common causes directly.
        # DoWhy model takes a single treatment typically for its main API, 
        # but multiple treatments can be passed as a list in some versions.
        # Here we define the graph explicitly.
        model = dowhy.CausalModel(
            data=df,
            treatment=treatment,
            outcome=self.outcome,
            graph=self.generate_gml()
        )
        return model

if __name__ == "__main__":
    # Example to visualize or verify the DAG
    dag = AirApportionmentDAG()
    print("Causal Graph GML:")
    print(dag.generate_gml())
