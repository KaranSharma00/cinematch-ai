"""
╔══════════════════════════════════════════════════════════════════╗
║              🎬 CINEMATCH AI - MOVIE RECOMMENDER                 ║
║        Complete Standalone Streamlit App (LOCAL POSTERS)         ║
║                                                                   ║
║  ✅ NO API KEY NEEDED - Uses local poster images!                ║
║  ✅ Smart poster matching (handles filename variations)          ║
║  ✅ Automatic fallback to gradient placeholders                  ║
║  ✅ Fixed deprecation warnings                                   ║
║                                                                   ║
║  Setup:                                                           ║
║  1. Ensure posters/ folder has movie images                      ║
║  2. Run: streamlit run app.py                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress any remaining deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

POSTERS_DIR = "posters"

st.set_page_config(
    page_title="CineMatch AI - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════════════
# MOVIE DATASET (100 popular movies)
# ═══════════════════════════════════════════════════════════════════

MOVIES_DATA = [
    {"title": "The Dark Knight", "overview": "Batman raises the stakes in his war on crime with the help of Lieutenant Jim Gordon and District Attorney Harvey Dent against the Joker who plunges Gotham City into anarchy", "genres": "Action Crime Drama Thriller", "keywords": "superhero dark hero batman joker crime vigilante gotham chaos", "cast": "Christian Bale Heath Ledger Aaron Eckhart", "director": "Christopher Nolan"},
    {"title": "Inception", "overview": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO in a mind-bending heist", "genres": "Action Adventure SciFi Thriller", "keywords": "dream heist subconscious mind reality layers architect totem", "cast": "Leonardo DiCaprio Joseph Gordon-Levitt Ellen Page", "director": "Christopher Nolan"},
    {"title": "Interstellar", "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival on a distant planet as Earth becomes uninhabitable", "genres": "Adventure Drama SciFi", "keywords": "space wormhole black hole survival future science time relativity", "cast": "Matthew McConaughey Anne Hathaway Jessica Chastain", "director": "Christopher Nolan"},
    {"title": "The Matrix", "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers in a simulated world", "genres": "Action SciFi", "keywords": "virtual reality simulation hacker artificial intelligence rebel red pill neo", "cast": "Keanu Reeves Laurence Fishburne Carrie-Anne Moss", "director": "Lana Wachowski"},
    {"title": "Avengers Endgame", "overview": "After the devastating events of Infinity War the Avengers assemble once more to reverse Thanos actions and restore balance to the universe through time travel", "genres": "Action Adventure SciFi Superhero", "keywords": "superhero team time travel infinity stones universe save avengers marvel", "cast": "Robert Downey Jr Chris Evans Scarlett Johansson", "director": "Anthony Russo"},
    {"title": "Iron Man", "overview": "Billionaire Tony Stark creates a powered armor suit to escape captivity and becomes the technologically advanced superhero Iron Man fighting terrorism", "genres": "Action Adventure SciFi Superhero", "keywords": "superhero armor technology billionaire weapons stark marvel", "cast": "Robert Downey Jr Gwyneth Paltrow Jeff Bridges", "director": "Jon Favreau"},
    {"title": "Thor", "overview": "The powerful but arrogant god Thor is cast out of Asgard to live among humans in Midgard where he must prove himself worthy of his hammer", "genres": "Action Adventure Fantasy Superhero", "keywords": "norse mythology god hero thunder hammer asgard marvel", "cast": "Chris Hemsworth Natalie Portman Tom Hiddleston", "director": "Kenneth Branagh"},
    {"title": "Spider-Man", "overview": "When bitten by a radioactive spider high school student Peter Parker gains spider-like abilities and becomes the superhero Spider-Man protecting New York", "genres": "Action Adventure SciFi Superhero", "keywords": "superhero school teenager radioactive spider web new york marvel", "cast": "Tobey Maguire Kirsten Dunst Willem Dafoe", "director": "Sam Raimi"},
    {"title": "The Godfather", "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son amidst a war among rival families", "genres": "Crime Drama", "keywords": "mafia family power crime honor loyalty betrayal italian", "cast": "Marlon Brando Al Pacino James Caan", "director": "Francis Ford Coppola"},
    {"title": "Goodfellas", "overview": "The story of Henry Hill and his life in the mob covering his relationship with his wife and his mob partners through the 1955 to 1980 period", "genres": "Crime Drama Biography", "keywords": "gangster crime mob money drugs murder italian mafia", "cast": "Ray Liotta Robert De Niro Joe Pesci", "director": "Martin Scorsese"},
    {"title": "Pulp Fiction", "overview": "The lives of two mob hitmen a boxer a gangster and his wife and a pair of diner bandits intertwine in four tales of violence and redemption", "genres": "Crime Drama Thriller", "keywords": "crime nonlinear violence drugs hitman redemption tarantino", "cast": "John Travolta Uma Thurman Samuel L Jackson", "director": "Quentin Tarantino"},
    {"title": "Fight Club", "overview": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much more sinister and revolutionary", "genres": "Drama Thriller", "keywords": "underground fight soap insomnia anarchy twist identity psychology", "cast": "Brad Pitt Edward Norton Helena Bonham Carter", "director": "David Fincher"},
    {"title": "The Shawshank Redemption", "overview": "Two imprisoned men bond over a number of years finding solace and eventual redemption through acts of common decency in a harsh prison environment", "genres": "Drama", "keywords": "prison friendship hope letter escape freedom banking", "cast": "Tim Robbins Morgan Freeman Bob Gunton", "director": "Frank Darabont"},
    {"title": "Forrest Gump", "overview": "The presidencies of Kennedy and Johnson the events of Vietnam Watergate and other historical events unfold through the perspective of an Alabama man with IQ 75", "genres": "Drama Romance Comedy", "keywords": "life journey war peace disability love inspirational history vietnam", "cast": "Tom Hanks Robin Wright Gary Sinise", "director": "Robert Zemeckis"},
    {"title": "Titanic", "overview": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious ill-fated RMS Titanic during its tragic maiden voyage", "genres": "Drama Romance Disaster", "keywords": "ship ocean love luxury disaster iceberg tragedy 1912", "cast": "Leonardo DiCaprio Kate Winslet Billy Zane", "director": "James Cameron"},
    {"title": "Avatar", "overview": "A paraplegic marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home", "genres": "Action Adventure SciFi Fantasy", "keywords": "alien planet nature environment military resources navi pandora", "cast": "Sam Worthington Zoe Saldana Sigourney Weaver", "director": "James Cameron"},
    {"title": "Jurassic Park", "overview": "A pragmatic paleontologist visiting an almost complete theme park is tasked with protecting a couple of kids after a power failure causes the dinosaurs to run loose", "genres": "Adventure SciFi Thriller", "keywords": "dinosaur theme park DNA island children escape science genetic", "cast": "Sam Neill Laura Dern Jeff Goldblum", "director": "Steven Spielberg"},
    {"title": "The Lion King", "overview": "Lion prince Simba and his father are targeted by his evil uncle who wants to ascend the throne himself in the African savanna", "genres": "Animation Adventure Drama Family", "keywords": "lion king pride jealousy evil uncle africa animals disney", "cast": "Matthew Broderick Jeremy Irons James Earl Jones", "director": "Roger Allers"},
    {"title": "Toy Story", "overview": "A little boy named Andy loves to be in his room playing with his toys especially a doll named Woody who leads other toys on adventures", "genres": "Animation Adventure Comedy Family", "keywords": "toys come alive friendship cowboy space ranger pixar disney", "cast": "Tom Hanks Tim Allen Don Rickles", "director": "John Lasseter"},
    {"title": "Finding Nemo", "overview": "After Nemo a young clownfish is captured and taken to Sydney his father Marlin must travel the entire ocean to find him with a forgetful fish friend", "genres": "Animation Adventure Comedy Family", "keywords": "fish ocean adventure father son clownfish reef pixar ocean", "cast": "Albert Brooks Ellen DeGeneres Alexander Gould", "director": "Andrew Stanton"},
    {"title": "Harry Potter and the Sorcerer's Stone", "overview": "Harry Potter a young wizard discovers his magical heritage on his eleventh birthday when he receives a letter of acceptance to Hogwarts School of Witchcraft", "genres": "Adventure Fantasy Family", "keywords": "magic school wizard wand spell potion owl hogwarts british", "cast": "Daniel Radcliffe Emma Watson Rupert Grint", "director": "Chris Columbus"},
    {"title": "The Lord of the Rings", "overview": "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron", "genres": "Adventure Fantasy Drama", "keywords": "ring quest evil dark lord fellowship journey hobbit middle earth", "cast": "Elijah Wood Ian McKellen Orlando Bloom", "director": "Peter Jackson"},
    {"title": "Star Wars", "overview": "Luke Skywalker a farm boy from a desert planet joins rebel forces and uses the Force to battle the evil Galactic Empire and rescue Princess Leia", "genres": "Action Adventure SciFi Fantasy", "keywords": "galaxy empire rebel force lightsaber jedi dark side space", "cast": "Mark Hamill Harrison Ford Carrie Fisher", "director": "George Lucas"},
    {"title": "Guardians of the Galaxy", "overview": "A group of intergalactic criminals must pull together to stop a fanatical warrior with plans to purge the universe using an ancient orb", "genres": "Action Adventure Comedy SciFi Superhero", "keywords": "space criminals team infinity stone raccoon groot marvel", "cast": "Chris Pratt Zoe Saldana Dave Bautista", "director": "James Gunn"},
    {"title": "Doctor Strange", "overview": "A brilliant but arrogant surgeon becomes a powerful sorcerer under the tutelage of the Ancient One after a career-ending car accident opens him to mysticism", "genres": "Action Adventure Fantasy SciFi Superhero", "keywords": "sorcerer magic dimension multiverse ancient one marvel mystic", "cast": "Benedict Cumberbatch Tilda Swinton Chiwetel Ejiofor", "director": "Scott Derrickson"},
    {"title": "Black Panther", "overview": "T'Challa returns home to the African nation of Wakanda to take his rightful place as king but must prevent a coup from destroying his hidden nation", "genres": "Action Adventure SciFi Superhero", "keywords": "africa king vibranium warrior tribe technology marvel wakanda", "cast": "Chadwick Boseman Michael B Jordan Lupita Nyong", "director": "Ryan Coogler"},
    {"title": "Wonder Woman", "overview": "When a pilot crashes and tells of conflict in the outside world Diana an Amazonian warrior leaves home to fight a war discovering her full powers and true destiny", "genres": "Action Adventure Fantasy Superhero", "keywords": "amazon warrior goddess war world enemy god dc greek", "cast": "Gal Gadot Chris Pine Robin Wright", "director": "Patty Jenkins"},
    {"title": "Aquaman", "overview": "Arthur Curry the human-born heir to the underwater kingdom of Atlantis goes on a quest to prevent a war between the worlds of ocean and land", "genres": "Action Adventure Fantasy Superhero", "keywords": "underwater ocean kingdom trident atlantis king dc atlantean", "cast": "Jason Momoa Amber Heard Nicole Kidman", "director": "James Wan"},
    {"title": "Captain America The First Avenger", "overview": "Steve Rogers a rejected military soldier transforms into Captain America after taking a dose of a Super-Soldier serum during World War II", "genres": "Action Adventure SciFi Superhero", "keywords": "soldier serum war brave shield america frozen marvel", "cast": "Chris Evans Hayley Atwell Sebastian Stan", "director": "Joe Johnston"},
    {"title": "Ant-Man", "overview": "Armed with a super-suit with the astonishing ability to shrink in scale but increase in strength Scott Lang must embrace his inner hero to pull off a daring heist", "genres": "Action Adventure Comedy SciFi Superhero", "keywords": "tiny suit shrink ant science criminal heist marvel", "cast": "Paul Rudd Michael Douglas Evangeline Lilly", "director": "Peyton Reed"},
    {"title": "Deadpool", "overview": "A wisecracking mercenary gets experimented on and becomes an immortal yet scarred mercenary with a warped sense of humor seeking revenge", "genres": "Action Comedy Superhero", "keywords": "mercenary humor fourth wall marvel violent comedy antihero", "cast": "Ryan Reynolds Morena Baccarin Ed Skrein", "director": "Tim Miller"},
    {"title": "Logan", "overview": "In a future where mutants are nearly extinct an aged and weary Logan cares for an ailing Professor X in a hideout on the Mexican border", "genres": "Action Drama SciFi Superhero", "keywords": "wolverine mutant future dying road trip marvel violent", "cast": "Hugh Jackman Patrick Stewart Dafne Keen", "director": "James Mangold"},
    {"title": "The Avengers", "overview": "Earth's mightiest heroes must come together and learn to fight as a team to stop the mischievous Loki and his alien army from enslaving humanity", "genres": "Action Adventure SciFi Superhero", "keywords": "superhero team loki aliens marvel avengers assemble", "cast": "Robert Downey Jr Chris Evans Mark Ruffalo", "director": "Joss Whedon"},
    {"title": "Gladiator", "overview": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery", "genres": "Action Drama Adventure", "keywords": "rome gladiator revenge emperor arena warrior historical", "cast": "Russell Crowe Joaquin Phoenix Connie Nielsen", "director": "Ridley Scott"},
    {"title": "Braveheart", "overview": "Scottish warrior William Wallace leads his countrymen in a rebellion to free his homeland from the tyranny of King Edward I of England", "genres": "Action Biography Drama War", "keywords": "scotland warrior rebellion freedom historical medieval war", "cast": "Mel Gibson Sophie Marceau Patrick McGoohan", "director": "Mel Gibson"},
    {"title": "Saving Private Ryan", "overview": "Following the Normandy Landings a group of US soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action", "genres": "Drama War Action", "keywords": "world war 2 normandy soldiers mission brothers historical", "cast": "Tom Hanks Matt Damon Tom Sizemore", "director": "Steven Spielberg"},
    {"title": "Schindler's List", "overview": "In German-occupied Poland during World War II industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution", "genres": "Biography Drama History War", "keywords": "holocaust nazi jewish savior poland historical world war", "cast": "Liam Neeson Ralph Fiennes Ben Kingsley", "director": "Steven Spielberg"},
    {"title": "The Silence of the Lambs", "overview": "A young FBI cadet must receive the help of an incarcerated cannibal killer to help catch another serial killer skinning his female victims", "genres": "Crime Drama Thriller Horror", "keywords": "serial killer fbi cannibal psychological thriller hannibal", "cast": "Jodie Foster Anthony Hopkins Scott Glenn", "director": "Jonathan Demme"},
    {"title": "Se7en", "overview": "Two detectives a rookie and a veteran hunt a serial killer who uses the seven deadly sins as his motives in a rain-soaked city", "genres": "Crime Drama Mystery Thriller", "keywords": "serial killer sins detective dark psychological thriller", "cast": "Morgan Freeman Brad Pitt Kevin Spacey", "director": "David Fincher"},
    {"title": "The Departed", "overview": "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston", "genres": "Crime Drama Thriller", "keywords": "undercover cop mole gang boston irish mafia crime", "cast": "Leonardo DiCaprio Matt Damon Jack Nicholson", "director": "Martin Scorsese"},
    {"title": "The Wolf of Wall Street", "overview": "Based on the true story of Jordan Belfort from his rise to a wealthy stock-broker living the high life to his fall involving crime corruption and the federal government", "genres": "Biography Crime Drama Comedy", "keywords": "wall street stocks fraud money crime excess drugs", "cast": "Leonardo DiCaprio Jonah Hill Margot Robbie", "director": "Martin Scorsese"},
    {"title": "Django Unchained", "overview": "With the help of a German bounty hunter a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner in the antebellum south", "genres": "Drama Western", "keywords": "slavery revenge bounty hunter western tarantino violence", "cast": "Jamie Foxx Christoph Waltz Leonardo DiCaprio", "director": "Quentin Tarantino"},
    {"title": "The Revenant", "overview": "A frontiersman on a fur trading expedition in the 1820s fights for survival after being mauled by a bear and left for dead by members of his own hunting team", "genres": "Adventure Drama Thriller", "keywords": "survival wilderness bear frontier revenge nature brutal", "cast": "Leonardo DiCaprio Tom Hardy Will Poulter", "director": "Alejandro Inarritu"},
    {"title": "Mad Max Fury Road", "overview": "In a post-apocalyptic wasteland a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners", "genres": "Action Adventure SciFi", "keywords": "post apocalyptic wasteland desert cars chase revolution", "cast": "Tom Hardy Charlize Theron Nicholas Hoult", "director": "George Miller"},
    {"title": "The Grand Budapest Hotel", "overview": "A writer encounters the owner of an aging high-class hotel who tells him of his early years serving as a lobby boy in the hotel's glorious years", "genres": "Adventure Comedy Drama", "keywords": "hotel european quirky whimsical adventure wes anderson", "cast": "Ralph Fiennes Tony Revolori Adrien Brody", "director": "Wes Anderson"},
    {"title": "La La Land", "overview": "While navigating their careers in Los Angeles a pianist and an actress fall in love but are faced with decisions that fray the fragile fabric of their romance", "genres": "Comedy Drama Music Romance", "keywords": "musical los angeles jazz dreams romance dancing", "cast": "Ryan Gosling Emma Stone John Legend", "director": "Damien Chazelle"},
    {"title": "Whiplash", "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential", "genres": "Drama Music", "keywords": "drums jazz music teacher perfectionism obsession conservatory", "cast": "Miles Teller JK Simmons Melissa Benoist", "director": "Damien Chazelle"},
    {"title": "Get Out", "overview": "A young African-American visits his white girlfriend's parents for the weekend where his simmering uneasiness about their reception of him eventually reaches a boiling point", "genres": "Horror Mystery Thriller", "keywords": "racism horror psychological family disturbing thriller", "cast": "Daniel Kaluuya Allison Williams Bradley Whitford", "director": "Jordan Peele"},
    {"title": "A Quiet Place", "overview": "In a post-apocalyptic world a family is forced to live in silence while hiding from monsters with ultra-sensitive hearing", "genres": "Drama Horror SciFi Thriller", "keywords": "silence monsters family survival post apocalyptic tension", "cast": "Emily Blunt John Krasinski Millicent Simmonds", "director": "John Krasinski"},
    {"title": "The Conjuring", "overview": "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse", "genres": "Horror Mystery Thriller", "keywords": "paranormal haunted house demons investigators exorcism", "cast": "Vera Farmiga Patrick Wilson Lili Taylor", "director": "James Wan"},
    {"title": "It", "overview": "In the summer of 1989 a group of bullied kids band together to destroy a shape-shifting monster which disguises itself as a clown and preys on the children of Derry", "genres": "Horror", "keywords": "clown children monster small town supernatural horror stephen king", "cast": "Bill Skarsgard Jaeden Martell Finn Wolfhard", "director": "Andy Muschietti"},
    {"title": "Joker", "overview": "In Gotham City mentally troubled comedian Arthur Fleck is disregarded and mistreated by society leading him to become the psychopathic criminal known as the Joker", "genres": "Crime Drama Thriller", "keywords": "joker mental illness gotham dark psychological origin villain", "cast": "Joaquin Phoenix Robert De Niro Zazie Beetz", "director": "Todd Phillips"},
    {"title": "1917", "overview": "April 1917 two young British soldiers are given a seemingly impossible mission to deliver a message deep in enemy territory that will stop 1600 men from walking into a deadly trap", "genres": "Drama War", "keywords": "world war 1 soldiers mission trenches british historical", "cast": "George MacKay Dean-Charles Chapman Mark Strong", "director": "Sam Mendes"},
    {"title": "Dunkirk", "overview": "Allied soldiers from Belgium the British Commonwealth and Empire and France are surrounded by the German Army and evacuated during a fierce battle in World War II", "genres": "Action Drama History Thriller War", "keywords": "world war 2 evacuation beach british soldiers historical", "cast": "Fionn Whitehead Tom Hardy Cillian Murphy", "director": "Christopher Nolan"},
    {"title": "The Prestige", "overview": "After a tragic accident two stage magicians in 1890s London engage in a bitter and increasingly-dangerous rivalry with fatal results", "genres": "Drama Mystery SciFi Thriller", "keywords": "magicians rivalry victorian london illusion mystery twist", "cast": "Christian Bale Hugh Jackman Michael Caine", "director": "Christopher Nolan"},
    {"title": "Memento", "overview": "A man with short-term memory loss attempts to track down his wife's murderer using notes and tattoos to hunt for the killer", "genres": "Mystery Thriller", "keywords": "memory loss revenge tattoos nonlinear puzzle psychological", "cast": "Guy Pearce Carrie-Anne Moss Joe Pantoliano", "director": "Christopher Nolan"},
    {"title": "The Truman Show", "overview": "An insurance salesman discovers his whole life is actually a reality TV show and everyone around him is an actor", "genres": "Comedy Drama SciFi", "keywords": "reality tv show life illusion identity meta philosophical", "cast": "Jim Carrey Ed Harris Laura Linney", "director": "Peter Weir"},
    {"title": "Eternal Sunshine of the Spotless Mind", "overview": "When their relationship turns sour a couple undergoes a medical procedure to have each other erased from their memories", "genres": "Drama Romance SciFi", "keywords": "memory erase relationship love romance surreal quirky", "cast": "Jim Carrey Kate Winslet Tom Wilkinson", "director": "Michel Gondry"},
    {"title": "Her", "overview": "In a near future a lonely writer develops an unlikely relationship with an operating system designed to meet his every need", "genres": "Drama Romance SciFi", "keywords": "artificial intelligence love loneliness future technology relationship", "cast": "Joaquin Phoenix Amy Adams Scarlett Johansson", "director": "Spike Jonze"},
    {"title": "Ex Machina", "overview": "A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a highly advanced humanoid AI", "genres": "Drama SciFi Thriller", "keywords": "artificial intelligence robot test isolation psychological thriller", "cast": "Alicia Vikander Domhnall Gleeson Oscar Isaac", "director": "Alex Garland"},
    {"title": "Blade Runner 2049", "overview": "Young Blade Runner K's discovery of a long buried secret leads him to track down former Blade Runner Rick Deckard who's been missing for thirty years", "genres": "Action Drama Mystery SciFi Thriller", "keywords": "replicant future dystopia noir detective visual sci-fi", "cast": "Ryan Gosling Harrison Ford Ana de Armas", "director": "Denis Villeneuve"},
    {"title": "Arrival", "overview": "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world", "genres": "Drama SciFi Mystery", "keywords": "aliens communication language linguistics first contact time", "cast": "Amy Adams Jeremy Renner Forest Whitaker", "director": "Denis Villeneuve"},
    {"title": "Dune", "overview": "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir becomes troubled by visions of a dark future", "genres": "Action Adventure Drama SciFi", "keywords": "desert planet spice noble family war visions space epic", "cast": "Timothee Chalamet Rebecca Ferguson Zendaya", "director": "Denis Villeneuve"},
    {"title": "The Social Network", "overview": "As Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook he is sued by the twins who claimed he stole their idea", "genres": "Biography Drama", "keywords": "facebook zuckerberg harvard social media lawsuit tech startup", "cast": "Jesse Eisenberg Andrew Garfield Justin Timberlake", "director": "David Fincher"},
    {"title": "Steve Jobs", "overview": "Steve Jobs takes us behind the scenes of the digital revolution to paint an intimate portrait of the brilliant man at its epicenter", "genres": "Biography Drama", "keywords": "apple technology innovation biography ceo computer revolutionary", "cast": "Michael Fassbender Kate Winslet Seth Rogen", "director": "Danny Boyle"},
    {"title": "The Imitation Game", "overview": "During World War II mathematician Alan Turing tries to crack the enigma code with help from fellow mathematicians while attempting to come to terms with his troubled private life", "genres": "Biography Drama Thriller War", "keywords": "world war 2 code breaking mathematician turing enigma", "cast": "Benedict Cumberbatch Keira Knightley Matthew Goode", "director": "Morten Tyldum"},
    {"title": "A Beautiful Mind", "overview": "After John Nash a brilliant but asocial mathematician accepts secret work in cryptography his life takes a turn for the nightmarish", "genres": "Biography Drama", "keywords": "mathematician schizophrenia genius mental illness biography nobel", "cast": "Russell Crowe Ed Harris Jennifer Connelly", "director": "Ron Howard"},
    {"title": "The Theory of Everything", "overview": "A look at the relationship between the famous physicist Stephen Hawking and his wife Jane as he faces motor neuron disease and rises to global renown", "genres": "Biography Drama Romance", "keywords": "stephen hawking physics als disease romance biography science", "cast": "Eddie Redmayne Felicity Jones Charlie Cox", "director": "James Marsh"},
    {"title": "Bohemian Rhapsody", "overview": "The story of the legendary British rock band Queen and lead singer Freddie Mercury leading up to their famous performance at Live Aid in 1985", "genres": "Biography Drama Music", "keywords": "queen freddie mercury rock music biography live aid band", "cast": "Rami Malek Lucy Boynton Gwilym Lee", "director": "Bryan Singer"},
    {"title": "Rocketman", "overview": "A musical fantasy about the fantastical human story of Elton John's breakthrough years chronicling his rise to fame with hits and personal struggles", "genres": "Biography Drama Music Fantasy", "keywords": "elton john music biography piano rock fame musical fantasy", "cast": "Taron Egerton Jamie Bell Richard Madden", "director": "Dexter Fletcher"},
    {"title": "A Star Is Born", "overview": "A musician helps a young singer find fame as age and alcoholism send his own career into a downward spiral in this modern remake", "genres": "Drama Music Romance", "keywords": "music romance fame alcoholism singer country rock star", "cast": "Bradley Cooper Lady Gaga Sam Elliott", "director": "Bradley Cooper"},
    {"title": "The Greatest Showman", "overview": "Celebrates the birth of show business and tells of a visionary who rose from nothing to create a spectacle that became a worldwide sensation", "genres": "Biography Drama Musical", "keywords": "circus musical show business barnum spectacle broadway singing", "cast": "Hugh Jackman Michelle Williams Zac Efron", "director": "Michael Gracey"},
    {"title": "Frozen", "overview": "When the newly-crowned Queen Elsa accidentally uses her power to turn things into ice to curse her home in infinite winter her sister Anna teams up with a mountain man and his reindeer to change the weather", "genres": "Animation Adventure Comedy Family Musical", "keywords": "sisters ice queen disney princess magic winter norwegian", "cast": "Kristen Bell Idina Menzel Josh Gad", "director": "Chris Buck"},
    {"title": "Moana", "overview": "In Ancient Polynesia when a terrible curse incurred by the Demigod Maui reaches Moana's island she answers the Ocean's call to seek out the Demigod to set things right", "genres": "Animation Adventure Comedy Family Musical", "keywords": "polynesian ocean adventure disney princess demigod tropical", "cast": "Auli'i Cravalho Dwayne Johnson Rachel House", "director": "Ron Clements"},
    {"title": "Coco", "overview": "Aspiring musician Miguel confronted with his family's ancestral ban on music enters the Land of the Dead to find his great-great-grandfather a legendary singer", "genres": "Animation Adventure Family Fantasy Music", "keywords": "mexican family music day of dead pixar guitar ancestors", "cast": "Anthony Gonzalez Gael Garcia Bernal Benjamin Bratt", "director": "Lee Unkrich"},
    {"title": "Up", "overview": "78-year-old Carl Fredricksen travels to Paradise Falls in his house equipped with balloons inadvertently taking a young stowaway along for the ride", "genres": "Animation Adventure Comedy Drama Family", "keywords": "elderly balloons house adventure pixar friendship south america", "cast": "Ed Asner Christopher Plummer Jordan Nagai", "director": "Pete Docter"},
    {"title": "WALL-E", "overview": "In the distant future a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind", "genres": "Animation Adventure Family SciFi", "keywords": "robot future earth space pixar environmental love silent", "cast": "Ben Burtt Elissa Knight Jeff Garlin", "director": "Andrew Stanton"},
    {"title": "Inside Out", "overview": "After young Riley is uprooted from her Midwest life and moved to San Francisco her emotions - Joy Fear Anger Disgust and Sadness - conflict on how best to navigate a new city house and school", "genres": "Animation Adventure Comedy Drama Family", "keywords": "emotions mind childhood pixar psychological family growing up", "cast": "Amy Poehler Bill Hader Lewis Black", "director": "Pete Docter"},
    {"title": "Shrek", "overview": "A mean lord exiles fairytale creatures to the swamp of a grumpy ogre who must go on a quest and rescue a princess for the lord in order to get his land back", "genres": "Animation Adventure Comedy Family Fantasy", "keywords": "ogre swamp princess dragon fairy tale comedy dreamworks", "cast": "Mike Myers Eddie Murphy Cameron Diaz", "director": "Andrew Adamson"},
    {"title": "How to Train Your Dragon", "overview": "A hapless young Viking who aspires to hunt dragons becomes the unlikely friend of a young dragon himself and learns there may be more to the creatures than he assumed", "genres": "Animation Adventure Family Fantasy", "keywords": "vikings dragon friendship dreamworks flying adventure norse", "cast": "Jay Baruchel Gerard Butler Christopher Mintz-Plasse", "director": "Dean DeBlois"},
    {"title": "Kung Fu Panda", "overview": "In the Valley of Peace Po the Panda finds himself chosen as the Dragon Warrior despite the fact that he is obese and a complete novice at martial arts", "genres": "Animation Action Adventure Comedy Family", "keywords": "panda martial arts china warrior dreamworks kung fu comedy", "cast": "Jack Black Angelina Jolie Dustin Hoffman", "director": "Mark Osborne"},
    {"title": "The Incredibles", "overview": "A family of undercover superheroes while trying to live the quiet suburban life are forced into action to save the world", "genres": "Animation Action Adventure Family", "keywords": "superhero family pixar suburban powers spy retro", "cast": "Craig T Nelson Holly Hunter Samuel L Jackson", "director": "Brad Bird"},
    {"title": "Zootopia", "overview": "In a city of anthropomorphic animals a rookie bunny cop and a cynical con artist fox must work together to uncover a conspiracy", "genres": "Animation Adventure Comedy Family Mystery", "keywords": "animals city police disney conspiracy detective anthropomorphic", "cast": "Ginnifer Goodwin Jason Bateman Idris Elba", "director": "Byron Howard"},
    {"title": "The Matrix Reloaded", "overview": "Freedom fighters Neo Trinity and Morpheus continue to lead the revolt against the Machine Army unleashing their arsenal of extraordinary skills and weaponry against the systematic forces of repression", "genres": "Action SciFi", "keywords": "matrix sequel virtual reality machine war revolution neo", "cast": "Keanu Reeves Laurence Fishburne Carrie-Anne Moss", "director": "Lana Wachowski"},
    {"title": "The Matrix Revolutions", "overview": "The human city of Zion defends itself against the massive invasion of the machines as Neo fights to end the war at another front while also opposing the rogue Agent Smith", "genres": "Action Adventure SciFi Thriller", "keywords": "matrix trilogy final battle zion machines neo smith", "cast": "Keanu Reeves Laurence Fishburne Carrie-Anne Moss", "director": "Lana Wachowski"},
    {"title": "The Hobbit", "overview": "A reluctant Hobbit Bilbo Baggins sets out to the Lonely Mountain with a spirited group of dwarves to reclaim their mountain home from Smaug the Dragon", "genres": "Adventure Fantasy", "keywords": "hobbit dwarves dragon mountain journey middle earth tolkien", "cast": "Martin Freeman Ian McKellen Richard Armitage", "director": "Peter Jackson"},
    {"title": "Pirates of the Caribbean", "overview": "Blacksmith Will Turner teams up with eccentric pirate Captain Jack Sparrow to save his love the governor's daughter from Jack's former pirate allies who are now undead", "genres": "Action Adventure Fantasy", "keywords": "pirates caribbean sea ship curse gold sword fighting", "cast": "Johnny Depp Orlando Bloom Keira Knightley", "director": "Gore Verbinski"},
    {"title": "Indiana Jones", "overview": "In 1936 archaeologist and adventurer Indiana Jones is hired by the US government to find the Ark of the Covenant before Adolf Hitler's Nazis can obtain its awesome powers", "genres": "Action Adventure", "keywords": "archaeologist adventure nazi ark whip hat treasure hunt", "cast": "Harrison Ford Karen Allen Paul Freeman", "director": "Steven Spielberg"},
    {"title": "Back to the Future", "overview": "Marty McFly a 17-year-old high school student is accidentally sent thirty years into the past in a time-traveling DeLorean invented by his close friend the eccentric scientist Doc Brown", "genres": "Adventure Comedy SciFi", "keywords": "time travel delorean 80s scientist high school past future", "cast": "Michael J Fox Christopher Lloyd Lea Thompson", "director": "Robert Zemeckis"},
    {"title": "E.T. the Extra-Terrestrial", "overview": "A troubled child summons the courage to help a friendly alien escape from Earth and return to his home planet", "genres": "Adventure Family SciFi", "keywords": "alien child friendship suburbia bike moon home spielberg", "cast": "Henry Thomas Drew Barrymore Peter Coyote", "director": "Steven Spielberg"},
    {"title": "Jaws", "overview": "When a killer shark unleashes chaos on a beach community it's up to a local sheriff a marine biologist and an old seafarer to hunt the beast down", "genres": "Adventure Thriller", "keywords": "shark ocean beach hunt fear boat summer classic", "cast": "Roy Scheider Robert Shaw Richard Dreyfuss", "director": "Steven Spielberg"},
    {"title": "The Shining", "overview": "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence while his psychic son sees horrific forebodings from both past and future", "genres": "Drama Horror", "keywords": "hotel haunted family isolation winter psychic kubrick stephen king", "cast": "Jack Nicholson Shelley Duvall Danny Lloyd", "director": "Stanley Kubrick"},
    {"title": "2001: A Space Odyssey", "overview": "After discovering a mysterious artifact buried beneath the Lunar surface mankind sets off on a quest to find its origins with help from intelligent supercomputer HAL 9000", "genres": "Adventure SciFi", "keywords": "space odyssey monolith computer artificial intelligence kubrick classic", "cast": "Keir Dullea Gary Lockwood William Sylvester", "director": "Stanley Kubrick"},
    {"title": "Alien", "overview": "After a space merchant vessel receives an unknown transmission as a distress call one of the crew is attacked by a mysterious life form and they soon realize that its life cycle has merely begun", "genres": "Horror SciFi", "keywords": "alien space horror monster ship claustrophobic ripley scary", "cast": "Sigourney Weaver Tom Skerritt John Hurt", "director": "Ridley Scott"},
    {"title": "The Terminator", "overview": "A human soldier is sent from 2029 to 1984 to stop an almost indestructible cyborg killing machine sent from the same year which has been programmed to execute a young woman", "genres": "Action SciFi", "keywords": "terminator cyborg time travel future killer machine classic", "cast": "Arnold Schwarzenegger Linda Hamilton Michael Biehn", "director": "James Cameron"},
    {"title": "Terminator 2 Judgment Day", "overview": "A cyborg identical to the one who failed to kill Sarah Connor must now protect her ten year old son John from a more advanced and powerful cyborg", "genres": "Action SciFi", "keywords": "terminator sequel cyborg protection future war t1000 liquid metal", "cast": "Arnold Schwarzenegger Linda Hamilton Edward Furlong", "director": "James Cameron"},
    {"title": "Predator", "overview": "A team of commandos on a mission in a Central American jungle find themselves hunted by an extraterrestrial warrior", "genres": "Action Adventure SciFi Thriller", "keywords": "alien jungle commandos hunter invisible action classic 80s", "cast": "Arnold Schwarzenegger Carl Weathers Kevin Peter Hall", "director": "John McTiernan"},
    {"title": "Die Hard", "overview": "An NYPD officer tries to save his wife and several others taken hostage by German terrorists during a Christmas party at the Nakatomi Plaza in Los Angeles", "genres": "Action Thriller", "keywords": "hostage building christmas terrorist cop action classic 80s", "cast": "Bruce Willis Alan Rickman Bonnie Bedelia", "director": "John McTiernan"},
    {"title": "John Wick", "overview": "An ex-hitman comes out of retirement to track down the gangsters that killed his dog and took everything from him", "genres": "Action Crime Thriller", "keywords": "hitman revenge dog assassin action gun fu stylish", "cast": "Keanu Reeves Michael Nyqvist Alfie Allen", "director": "Chad Stahelski"},
    {"title": "Mission Impossible Fallout", "overview": "Ethan Hunt and his IMF team along with some familiar allies race against time after a mission gone wrong to prevent nuclear disaster", "genres": "Action Adventure Thriller", "keywords": "spy mission impossible stunts action nuclear tom cruise", "cast": "Tom Cruise Henry Cavill Ving Rhames", "director": "Christopher McQuarrie"},
    {"title": "Casino Royale", "overview": "After earning 00 status and a licence to kill Secret Agent James Bond sets out on his first mission as 007 to defeat a private banker funding terrorists in a high stakes game of poker", "genres": "Action Adventure Thriller", "keywords": "james bond 007 spy casino poker terrorism british", "cast": "Daniel Craig Eva Green Mads Mikkelsen", "director": "Martin Campbell"},
    {"title": "Skyfall", "overview": "James Bond's loyalty to M is tested when her past comes back to haunt her Whilst MI6 comes under attack 007 must track down and destroy the threat no matter how personal the cost", "genres": "Action Adventure Thriller", "keywords": "james bond 007 spy british mi6 personal past scotland", "cast": "Daniel Craig Judi Dench Javier Bardem", "director": "Sam Mendes"},
]


# ═══════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'his', 'her', 'their', 'our', 'my', 'your'
}


def simple_stem(word):
    """Simple stemming without NLTK."""
    suffixes = ['ing', 'ly', 'ed', 'es', 's', 'er', 'est', 'ion', 'tion']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def clean_text(text):
    """Clean text for vectorization."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [simple_stem(w) for w in words if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)


# ═══════════════════════════════════════════════════════════════════
# 🎬 SMART POSTER LOADING (Handles ALL filename variations!)
# ═══════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.JPG', '.JPEG', '.PNG']


def generate_filename_variations(movie_title):
    """Generate ALL possible filename variations for a movie title."""
    variations = []
    
    variations.append(movie_title)
    variations.append(movie_title.replace(' ', '_'))
    
    no_apostrophe = movie_title.replace("'", "").replace(" ", "_")
    variations.append(no_apostrophe)
    
    cleaned = "".join(c for c in movie_title if c.isalnum() or c in (' ', '-', '_')).strip()
    variations.append(cleaned.replace(' ', '_'))
    
    no_hyphen = movie_title.replace('-', '').replace(' ', '_')
    variations.append(no_hyphen)
    
    hyphen_to_under = movie_title.replace('-', '_').replace(' ', '_')
    variations.append(hyphen_to_under)
    
    variations.append(movie_title.lower().replace(' ', '_'))
    variations.append(cleaned.lower().replace(' ', '_'))
    
    no_period = movie_title.replace('.', '').replace(' ', '_')
    variations.append(no_period)
    
    if 'E.T.' in movie_title or 'E.T' in movie_title:
        variations.append('ET_the_ExtraTerrestrial')
        variations.append('ET_the_Extra_Terrestrial')
        variations.append('E_T_the_Extra_Terrestrial')
    
    # Remove duplicates
    seen = set()
    unique_variations = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)
    
    return unique_variations


@st.cache_data
def build_poster_index():
    """Build an index of all poster files."""
    if not os.path.exists(POSTERS_DIR):
        return {}
    
    poster_index = {}
    
    for filename in os.listdir(POSTERS_DIR):
        filepath = os.path.join(POSTERS_DIR, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        name_without_ext = os.path.splitext(filename)[0]
        
        poster_index[name_without_ext] = filepath
        poster_index[name_without_ext.lower()] = filepath
        
        clean = "".join(c for c in name_without_ext if c.isalnum() or c == '_').lower()
        poster_index[clean] = filepath
        
        with_spaces = name_without_ext.replace('_', ' ')
        poster_index[with_spaces.lower()] = filepath
    
    return poster_index


def get_poster_path(movie_title):
    """Find the local poster file for a movie using smart matching."""
    if not os.path.exists(POSTERS_DIR):
        return None
    
    poster_index = build_poster_index()
    variations = generate_filename_variations(movie_title)
    
    for variation in variations:
        for ext in IMAGE_EXTENSIONS:
            filepath = os.path.join(POSTERS_DIR, variation + ext)
            if os.path.exists(filepath):
                return filepath
        
        if variation in poster_index:
            return poster_index[variation]
        
        if variation.lower() in poster_index:
            return poster_index[variation.lower()]
        
        clean = "".join(c for c in variation if c.isalnum() or c == '_').lower()
        if clean in poster_index:
            return poster_index[clean]
    
    return None


def get_gradient_placeholder(title, index):
    """Beautiful gradient placeholder when poster doesn't exist."""
    gradients = [
        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
        "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
        "linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%)",
        "linear-gradient(135deg, #2E3192 0%, #1BFFFF 100%)",
        "linear-gradient(135deg, #D4145A 0%, #FBB03B 100%)",
        "linear-gradient(135deg, #662D8C 0%, #ED1E79 100%)",
    ]
    gradient = gradients[index % len(gradients)]
    first_letter = title[0].upper() if title else "?"
    
    return f"""
    <div style="
        width: 100%;
        aspect-ratio: 2/3;
        background: {gradient};
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    ">
        <div style="
            font-size: 5rem;
            font-weight: 900;
            color: rgba(255,255,255,0.95);
            font-family: 'Cinzel', serif;
            text-shadow: 0 4px 20px rgba(0,0,0,0.5);
        ">{first_letter}</div>
        <div style="
            font-size: 0.65rem;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 0.8rem;
            padding: 0.3rem 0.8rem;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 15px;
        ">NO POSTER</div>
    </div>
    """


def display_poster(movie_title, index=0):
    """
    Display a movie poster - either the local file or a gradient placeholder.
    Uses the modern use_container_width parameter (no deprecation warnings).
    """
    poster_path = get_poster_path(movie_title)
    
    if poster_path:
        try:
            # ✅ FIXED: Using use_container_width instead of deprecated use_column_width
            st.image(poster_path, width='stretch')
            return True
        except Exception:
            st.markdown(
                get_gradient_placeholder(movie_title, index),
                unsafe_allow_html=True
            )
            return False
    else:
        st.markdown(
            get_gradient_placeholder(movie_title, index),
            unsafe_allow_html=True
        )
        return False


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATION SYSTEM
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def build_recommendation_system():
    """Build DataFrame + both similarity matrices."""
    df = pd.DataFrame(MOVIES_DATA)
    df['movie_id'] = range(1, len(df) + 1)
    
    df['tags'] = (
        df['overview'] + ' ' + df['genres'] + ' ' + 
        df['keywords'] + ' ' + df['cast'] + ' ' + df['director']
    )
    df['tags'] = df['tags'].apply(clean_text)
    
    cv = CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    count_matrix = cv.fit_transform(df['tags'])
    count_similarity = cosine_similarity(count_matrix)
    
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    tfidf_similarity = cosine_similarity(tfidf_matrix)
    
    return df, count_similarity, tfidf_similarity


def get_recommendations(df, similarity_matrix, movie_title, n=5):
    """Get top N similar movies."""
    matches = df[df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return []
    
    idx = matches.index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    return [{
        'title': df.iloc[i]['title'],
        'score': float(score),
        'genres': df.iloc[i]['genres'],
        'director': df.iloc[i]['director'],
        'cast': df.iloc[i]['cast'],
        'overview': df.iloc[i]['overview'][:150] + '...'
    } for i, score in scores]


def count_available_posters(df):
    """Count how many movies have local posters available."""
    count = 0
    for title in df['title']:
        if get_poster_path(title):
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════
# CUSTOM CSS (Dark Cinema Theme with Glassmorphism)
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Raleway:wght@300;400;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0A0A0A 0%, #111111 50%, #0D0D1A 100%);
    background-attachment: fixed;
}

#MainMenu, footer, header { visibility: hidden; }

.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    background: linear-gradient(180deg, rgba(245,166,35,0.08) 0%, transparent 100%);
    border-bottom: 1px solid rgba(245,166,35,0.3);
    margin-bottom: 2rem;
    animation: fadeInDown 1s ease-out;
}

.main-title {
    font-family: 'Cinzel', serif;
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #F5A623 0%, #F7C948 50%, #E8891A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: 3px;
    text-shadow: 0 0 40px rgba(245,166,35,0.3);
}

.main-subtitle {
    font-family: 'Raleway', sans-serif;
    font-size: 1.1rem;
    color: rgba(255,255,255,0.6);
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.film-strip {
    font-size: 1.2rem;
    letter-spacing: 8px;
    color: rgba(245,166,35,0.3);
    margin-top: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #F5A623 0%, #E8891A 100%) !important;
    color: #0A0A0A !important;
    font-family: 'Raleway', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 2.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(245,166,35,0.4) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(245,166,35,0.6) !important;
}

.stSelectbox > div > div {
    background: rgba(20, 20, 20, 0.9) !important;
    border: 1px solid rgba(245,166,35,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}

.selected-card {
    background: rgba(28, 28, 28, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(245,166,35,0.3);
    border-left: 4px solid #F5A623;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    animation: slideInLeft 0.6s ease-out;
}

.selected-label {
    font-family: 'Raleway', sans-serif;
    font-size: 0.75rem;
    color: #F5A623;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 600;
}

.selected-title {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    color: white;
    margin: 0.3rem 0 0.5rem 0;
}

.selected-meta {
    color: rgba(255,255,255,0.6);
    font-family: 'Raleway', sans-serif;
    font-size: 0.9rem;
    line-height: 1.8;
}

.movie-card {
    background: rgba(20, 20, 20, 0.85);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 1rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 1rem;
    animation: fadeInUp 0.6s ease-out;
}

.movie-card:hover {
    border-color: rgba(245,166,35,0.5);
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 40px rgba(0,0,0,0.6), 0 0 30px rgba(245,166,35,0.2);
}

.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #F5A623, #E8891A);
    color: #0A0A0A;
    font-weight: 800;
    font-size: 0.75rem;
    padding: 0.3rem 0.7rem;
    border-radius: 15px;
    font-family: 'Raleway', sans-serif;
    box-shadow: 0 4px 12px rgba(245,166,35,0.5);
    margin-bottom: 0.5rem;
}

.movie-card-title {
    font-family: 'Raleway', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: white;
    margin: 0.8rem 0 0.5rem 0;
    min-height: 2.8rem;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1.3;
}

.sim-bar-container {
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    height: 6px;
    margin: 0.6rem 0;
    overflow: hidden;
}

.sim-bar-fill {
    background: linear-gradient(90deg, #E8891A, #F5A623, #F7C948);
    height: 100%;
    border-radius: 10px;
    transition: width 1.5s ease;
    box-shadow: 0 0 10px rgba(245,166,35,0.5);
}

.similarity-score {
    font-family: 'Raleway', sans-serif;
    font-size: 0.8rem;
    color: #F5A623;
    font-weight: 600;
    letter-spacing: 1px;
}

.section-header {
    font-family: 'Cinzel', serif;
    font-size: 1.6rem;
    color: white;
    margin: 2rem 0 1.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(245,166,35,0.3);
    position: relative;
}

.section-header::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 80px;
    height: 2px;
    background: #F5A623;
    box-shadow: 0 0 10px #F5A623;
}

[data-testid="stSidebar"] {
    background: rgba(10, 10, 10, 0.95) !important;
    border-right: 1px solid rgba(245,166,35,0.2) !important;
}

.sidebar-logo {
    text-align: center;
    padding: 1.2rem 0;
    border-bottom: 1px solid rgba(245,166,35,0.2);
    margin-bottom: 1rem;
}

.sidebar-logo-text {
    font-family: 'Cinzel', serif;
    font-size: 1.7rem;
    background: linear-gradient(135deg, #F5A623, #F7C948);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

.sidebar-subtitle {
    color: rgba(255,255,255,0.4);
    font-size: 0.65rem;
    letter-spacing: 3px;
    font-family: 'Raleway', sans-serif;
    margin-top: 0.3rem;
}

.sidebar-section {
    font-family: 'Cinzel', serif;
    color: #F5A623;
    font-size: 0.9rem;
    margin: 1.2rem 0 0.6rem 0;
    letter-spacing: 2px;
}

.step-card {
    background: rgba(245,166,35,0.05);
    border: 1px solid rgba(245,166,35,0.15);
    border-radius: 8px;
    padding: 0.7rem;
    margin: 0.4rem 0;
    font-family: 'Raleway', sans-serif;
    color: rgba(255,255,255,0.8);
    font-size: 0.8rem;
}

.step-number {
    color: #F5A623;
    font-weight: 700;
}

.metric-card {
    background: rgba(245,166,35,0.05);
    border: 1px solid rgba(245,166,35,0.15);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F5A623;
    font-family: 'Raleway', sans-serif;
}

.metric-label {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'Raleway', sans-serif;
    margin-top: 0.2rem;
}

.error-card {
    background: rgba(220, 38, 38, 0.1);
    border: 1px solid rgba(220, 38, 38, 0.3);
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    color: white;
}

.tech-badge {
    display: inline-block;
    background: rgba(245,166,35,0.1);
    border: 1px solid rgba(245,166,35,0.3);
    color: #F5A623;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin: 0.3rem;
    font-size: 0.75rem;
    font-family: 'Raleway', sans-serif;
    font-weight: 600;
}

.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(245,166,35,0.2);
    font-family: 'Raleway', sans-serif;
    color: rgba(255,255,255,0.4);
    font-size: 0.85rem;
}

.stSlider > div > div > div > div {
    background: #F5A623 !important;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

@media (max-width: 768px) {
    .main-title { font-size: 2.5rem; }
    .main-subtitle { font-size: 0.9rem; letter-spacing: 3px; }
    .selected-title { font-size: 1.4rem; }
    .section-header { font-size: 1.3rem; }
}
</style>
"""


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

def render_sidebar(df):
    """Render sidebar with poster statistics."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-text">🎬 CineMatch AI</div>
            <div class="sidebar-subtitle">MOVIE RECOMMENDER</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Poster status
        st.markdown('<div class="sidebar-section">🖼️ POSTER STATUS</div>', 
                    unsafe_allow_html=True)
        
        if os.path.exists(POSTERS_DIR):
            matched_count = count_available_posters(df)
            total = len(df)
            
            if matched_count == total:
                st.success(f"✅ All {total} posters loaded!")
            elif matched_count > 0:
                st.info(f"✅ {matched_count}/{total} posters found")
                st.caption("Missing use gradient placeholders")
            else:
                st.warning("⚠️ No matching posters found")
        else:
            st.error("❌ Posters folder not found")
        
        # How It Works
        st.markdown('<div class="sidebar-section">⚙️ HOW IT WORKS</div>', 
                    unsafe_allow_html=True)
        
        steps = [
            ("1", "Select a movie you enjoy"),
            ("2", "Choose recommendation count"),
            ("3", "Pick your ML algorithm"),
            ("4", "Click 'Recommend' & explore")
        ]
        
        for num, desc in steps:
            st.markdown(f"""
            <div class="step-card">
                <span class="step-number">Step {num}:</span> {desc}
            </div>
            """, unsafe_allow_html=True)
        
        # Algorithms
        st.markdown('<div class="sidebar-section">🧠 ALGORITHMS</div>', 
                    unsafe_allow_html=True)
        
        algo_df = pd.DataFrame({
            'Feature': ['Method', 'Speed', 'Accuracy'],
            'CountVec': ['Word Count', 'Faster', 'Good'],
            'TF-IDF': ['Weighted', 'Slower', 'Better']
        })
        st.dataframe(algo_df.set_index('Feature'), width='stretch')
        
        # Stats
        st.markdown('<div class="sidebar-section">📊 STATISTICS</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Movies</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            matched = count_available_posters(df) if os.path.exists(POSTERS_DIR) else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{matched}</div>
                <div class="metric-label">Posters</div>
            </div>
            """, unsafe_allow_html=True)
        
        # About
        st.markdown('<div class="sidebar-section">ℹ️ ABOUT</div>', 
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: Raleway; font-size: 0.8rem; 
                    color: rgba(255,255,255,0.6); line-height: 1.6;">
            Content-based recommender using cosine similarity. 
            All posters stored locally — no API needed!
        </div>
        """, unsafe_allow_html=True)
        
        # Missing posters (debug)
        if os.path.exists(POSTERS_DIR):
            missing = [t for t in df['title'] if not get_poster_path(t)]
            if missing:
                with st.expander(f"🔍 View {len(missing)} missing posters"):
                    for m in missing:
                        st.caption(f"• {m}")


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    
    # Load custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Build recommendation system (cached)
    with st.spinner("🎬 Loading CineMatch AI..."):
        df, count_sim, tfidf_sim = build_recommendation_system()
    
    # Render sidebar
    render_sidebar(df)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎬 CineMatch AI</h1>
        <p class="main-subtitle">Discover Your Next Favorite Film</p>
        <div class="film-strip">🎞️ ▶ ■ ▶ ■ ▶ ■ ▶ 🎞️</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search Section
    st.markdown("### 🔍 Find Your Movie")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        movie_list = [''] + sorted(df['title'].tolist())
        selected = st.selectbox("Select a movie you love...", options=movie_list)
    
    with col2:
        n_recs = st.slider("Recommendations", 3, 10, 5)
    
    with col3:
        algorithm = st.radio("Algorithm", ["CountVec", "TF-IDF"])
    
    # Recommend Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        recommend = st.button("🎯 GET RECOMMENDATIONS")
    
    # Process Recommendations
    if recommend:
        if not selected:
            st.markdown("""
            <div class="error-card">
                <h3>⚠️ Please select a movie first!</h3>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Pick similarity matrix
        sim_matrix = tfidf_sim if algorithm == "TF-IDF" else count_sim
        selected_movie = df[df['title'] == selected].iloc[0]
        
        # Selected movie card with poster
        col_poster, col_info = st.columns([1, 3])
        
        with col_poster:
            display_poster(selected, index=0)
        
        with col_info:
            st.markdown(f"""
            <div class="selected-card">
                <div class="selected-label">🎬 Based On Your Selection</div>
                <div class="selected-title">{selected}</div>
                <div class="selected-meta">
                    🎭 <strong>{selected_movie['genres']}</strong><br/>
                    🎬 Directed by <strong>{selected_movie['director']}</strong><br/>
                    ⭐ Starring: {selected_movie['cast']}
                </div>
                <div style="margin-top: 0.8rem; color: rgba(255,255,255,0.5); 
                            font-size: 0.85rem;">
                    Using <strong style="color: #F5A623">{algorithm}</strong> · 
                    Showing <strong style="color: #F5A623">{n_recs}</strong> matches
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Get recommendations
        with st.spinner("🎬 Finding perfect matches..."):
            time.sleep(0.3)
            recs = get_recommendations(df, sim_matrix, selected, n_recs)
        
        # Section Header
        st.markdown('<div class="section-header">🏆 Movies You\'ll Love</div>', 
                    unsafe_allow_html=True)
        
        # Display recommendations in rows of 5
        for row_start in range(0, len(recs), 5):
            row_recs = recs[row_start:row_start + 5]
            cols = st.columns(len(row_recs))
            
            for i, (col, rec) in enumerate(zip(cols, row_recs)):
                rank = row_start + i + 1
                sim_pct = int(rec['score'] * 100)
                
                with col:
                    # Display poster (auto-fallback to gradient)
                    display_poster(rec['title'], index=rank - 1)
                    
                    # Info card
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="rank-badge">#{rank}</div>
                        <div class="movie-card-title">{rec['title']}</div>
                        <div class="sim-bar-container">
                            <div class="sim-bar-fill" style="width: {sim_pct}%"></div>
                        </div>
                        <div class="similarity-score">★ {sim_pct}% MATCH</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Details expanders
        st.markdown("<br/>", unsafe_allow_html=True)
        
        with st.expander("📊 View Detailed Data"):
            detail_data = pd.DataFrame([{
                'Rank': f"#{i+1}",
                'Title': r['title'],
                'Genres': r['genres'],
                'Director': r['director'],
                'Match %': f"{int(r['score'] * 100)}%",
                'Has Poster': '✅' if get_poster_path(r['title']) else '❌'
            } for i, r in enumerate(recs)])
            st.dataframe(detail_data.set_index('Rank'), width='stretch')
        
        with st.expander("🎬 View Overviews"):
            for i, rec in enumerate(recs):
                st.markdown(f"""
                **#{i+1} — {rec['title']}** ({int(rec['score']*100)}% match)  
                *{rec['genres']}* · Directed by {rec['director']}  
                > {rec['overview']}
                """)
                st.markdown("---")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Python, Streamlit & Machine Learning</p>
        <p>
            <span class="tech-badge">Python</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Scikit-learn</span>
            <span class="tech-badge">Pandas</span>
            <span class="tech-badge">Local Posters</span>
        </p>
        <p style="margin-top: 1rem; font-size: 0.75rem;">
            🖼️ All posters loaded locally — 100% offline capable!
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# RUN APP
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()