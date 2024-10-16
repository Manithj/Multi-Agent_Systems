"""Module for simulating tourist behavior and collecting place information."""

import wikipediaapi
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

USER_AGENT = 'TouristPlaceInfoAgent/1.0'
WIKI_WIKI = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language='en')

def get_wikipedia_summary(place_name):
    """Retrieve Wikipedia summary for a given place."""
    page = WIKI_WIKI.page(place_name)
    if page.exists():
        return page.summary
    return f"Wikipedia page for {place_name} not found."

class TouristAgent(Agent):
    """Agent representing a tourist visiting places and collecting information."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.places_visited = []
        self.information_collected = {}

    def step(self):
        """Perform a single step of the agent's behavior."""
        if self.model.places:
            place = self.random.choice(self.model.places)
            if place not in self.places_visited:
                self.places_visited.append(place)
                self.information_collected[place] = get_wikipedia_summary(place)
                self.model.places.remove(place)

class TourismModel(Model):
    """Model simulating tourist behavior across multiple agents."""

    def __init__(self, num_agents, places):
        super().__init__()
        self.num_agents = num_agents
        self.places = places.copy()
        self.schedule = RandomActivation(self)

        for i in range(self.num_agents):
            agent = TouristAgent(i, self)
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            model_reporters={"Places_Remaining": lambda m: len(m.places)},
            agent_reporters={"Places_Visited": lambda a: len(a.places_visited)}
        )

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

def update_csv(data, csv_file):
    """Update or append data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Information has been added to {csv_file}")

def run_tourism_model(num_agents, places, num_steps):
    """Run the tourism model and collect data."""
    model = TourismModel(num_agents, places)
    for _ in range(num_steps):
        model.step()

    all_data = []
    for agent in model.schedule.agents:
        for place, info in agent.information_collected.items():
            all_data.append({
                "Agent": agent.unique_id,
                "Place": place,
                "Information": info
            })

    return all_data

if __name__ == "__main__":
    tourist_places = [
        "Eiffel Tower", "Taj Mahal", "Great Wall of China", "Machu Picchu",
        "Colosseum", "Statue of Liberty", "Christ the Redeemer", "Petra",
        "Great Barrier Reef", "Angkor Wat", "Sigiriya", "Temple of the Tooth",
        "Yala National Park", "Galle Fort", "Polonnaruwa", "Dambulla Cave Temple",
        "Nuwara Eliya", "Horton Plains National Park", "Mirissa Beach", "Adam's Peak"
    ]

    CSV_FILE = "tourist_places_info_mesa.csv"

    collected_data = run_tourism_model(num_agents=7, places=tourist_places, num_steps=30)
    update_csv(collected_data, CSV_FILE)

    print(f"Simulation complete. Data stored in {CSV_FILE}")