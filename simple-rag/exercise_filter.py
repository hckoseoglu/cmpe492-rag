import pandas as pd
from typing import List, Dict, Optional


# TODO instead of normalizing properly format the database (the excel file)


def normalize_muscle_name(muscle: str) -> str:
    """Normalize muscle names to match database format."""
    muscle_mapping = {
        "chest": "Chest",
        "back": "Back",
        "legs": "Quadriceps",  # Can also filter for Hamstrings, Calves
        "shoulders": "Shoulders",
        "arms": "Biceps",  # Can also filter for Triceps
        "core": "Abdominals",
        "abs": "Abdominals",
        "glutes": "Glutes",
        "biceps": "Biceps",
        "triceps": "Triceps",
        "quadriceps": "Quadriceps",
        "hamstrings": "Hamstrings",
        "calves": "Calves",
        "forearms": "Forearms",
        "trapezius": "Trapezius",
        "abductors": "Abductors",
        "adductors": "Adductors",
        "hip flexors": "Hip Flexors",
    }
    return muscle_mapping.get(muscle.lower(), muscle.title())


def normalize_difficulty(difficulty: str) -> str:
    """Normalize difficulty levels to match database format."""
    difficulty_mapping = {
        "beginner": "Beginner",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
        "novice": "Novice",
        "expert": "Expert",
        "master": "Master",
        "grand master": "Grand Master",
        "legendary": "Legendary",
    }
    return difficulty_mapping.get(difficulty.lower(), difficulty.title())


def normalize_equipment(equipment: str) -> str:
    """Normalize equipment names to match database format."""
    equipment_mapping = {
        "none": "Bodyweight",
        "bodyweight": "Bodyweight",
        "dumbbells": "Dumbbell",
        "dumbbell": "Dumbbell",
        "barbell": "Barbell",
        "resistance bands": "Miniband",
        "resistance band": "Miniband",
        "cable": "Cable",
        "gym": None,  # Don't filter by equipment for gym
    }
    return equipment_mapping.get(equipment.lower(), equipment.title())


def filter_exercises(
    excel_path: str,
    target_muscle: str,
    difficulty: str,
    equipment: str = "Bodyweight",
    num_exercises: int = 3,
) -> List[Dict]:
    """
    Filter exercises from Excel database based on criteria.

    Args:
        excel_path: Path to the exercise database Excel file
        target_muscle: Target muscle group (e.g., 'chest', 'back', 'legs')
        difficulty: Difficulty level (e.g., 'beginner', 'intermediate', 'advanced')
        equipment: Available equipment (e.g., 'none', 'dumbbells', 'barbell')
        num_exercises: Number of exercises to return

    Returns:
        List of exercise dictionaries with name, difficulty, muscle, equipment
    """
    # Load the Excel file
    df = pd.read_excel(excel_path)

    # Normalize inputs
    muscle = normalize_muscle_name(target_muscle)
    diff = normalize_difficulty(difficulty)
    equip = normalize_equipment(equipment)

    # Start filtering
    filtered_df = df.copy()

    # Filter by target muscle (handle the trailing space in column name)
    filtered_df = filtered_df[filtered_df["Target Muscle Group "].str.strip() == muscle]

    # Filter by difficulty # TODO create proper logic
    if diff == "Advanced":
        # For advanced users, include both "Advanced", "Intermediate" and "Beginner" exercises
        filtered_df = filtered_df[
            filtered_df["Difficulty Level"].isin(
                [
                    "Advanced",
                    "Intermediate",
                    "Beginner",
                    "Novice",
                    "Expert",
                    "Master",
                    "Grand Master",
                    "Legendary",
                ]
            )
        ]
    else:
        filtered_df = filtered_df[filtered_df["Difficulty Level"] == diff]

    # Filter by equipment (only if specific equipment requested)
    if equip is not None:
        filtered_df = filtered_df[
            filtered_df["Primary Equipment "].str.strip() == equip
        ]

    # Limit to requested number
    print("Number of exercises to include:", num_exercises)
    filtered_df = filtered_df.head(num_exercises)

    # Convert to list of dictionaries
    exercises = []
    for _, row in filtered_df.iterrows():
        exercises.append(
            {
                "name": row["Exercise"],
                "difficulty": row["Difficulty Level"],
                "target_muscle": row["Target Muscle Group "].strip(),
                "equipment": row["Primary Equipment "].strip(),
                "prime_mover": row["Prime Mover Muscle"],
                "secondary_muscle": row["Secondary Muscle"],
                "mechanics": row["Mechanics"],
                "force": row["Force Type"],
            }
        )

    return exercises


# Example usage
if __name__ == "__main__":
    excel_path = "./exercise_db.xlsx"

    print("=" * 60)
    print("EXAMPLE 1: Chest exercises for beginners (no equipment)")
    print("=" * 60)

    exercises = filter_exercises(
        excel_path=excel_path,
        target_muscle="chest",
        difficulty="beginner",
        equipment="none",
        num_exercises=3,
    )

    print(f"\n✅ Found {len(exercises)} exercises:\n")
    for i, ex in enumerate(exercises, 1):
        print(f"{i}. {ex['name']}")
        print(f"   Difficulty: {ex['difficulty']}")
        print(f"   Equipment: {ex['equipment']}")
        print(f"   Prime Mover: {ex['prime_mover']}")
        print(f"   YouTube Demo: {ex['youtube_demo']}\n")

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Shoulder exercises with dumbbells")
    print("=" * 60)

    exercises = filter_exercises(
        excel_path=excel_path,
        target_muscle="shoulders",
        difficulty="beginner",
        equipment="dumbbells",
        num_exercises=5,
    )

    print(f"\n✅ Found {len(exercises)} exercises:\n")
    for i, ex in enumerate(exercises, 1):
        print(f"{i}. {ex['name']}")
        print(f"   Difficulty: {ex['difficulty']}")
        print(f"   Equipment: {ex['equipment']}\n")

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Back exercises for intermediate")
    print("=" * 60)

    exercises = filter_exercises(
        excel_path=excel_path,
        target_muscle="back",
        difficulty="intermediate",
        equipment="none",
        num_exercises=4,
    )

    print(f"\n✅ Found {len(exercises)} exercises:\n")
    for i, ex in enumerate(exercises, 1):
        print(f"{i}. {ex['name']}")
        print(f"   Difficulty: {ex['difficulty']}")
