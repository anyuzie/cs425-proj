"""Generates the O*NET-independent data: 90 synthetic seeds, question bank, 60 NLI pairs.
LLM-drafted; every line still needs a human review before use."""
import json, random, os
random.seed(425)
OUT = "/home/claude/data/out"; os.makedirs(OUT, exist_ok=True)
T = ["R", "I", "A", "S", "E", "C"]
def slots(**kw): return {t: kw.get(t, "neutral") for t in T}

# 90 synthetic seeds
ENJ = [  # (text, type, polarity); 5 per type, mixed sign, reason = enjoyment
 ("I love building things with my hands, like fixing up my bike on weekends.", "R", "positive"),
 ("Being outdoors and physically doing something all day sounds great to me.", "R", "positive"),
 ("I really enjoy assembling furniture and figuring out how the parts fit.", "R", "positive"),
 ("Working with tools and machines just doesn't interest me at all.", "R", "negative"),
 ("I'd hate a job where I'm on my feet doing manual work.", "R", "negative"),
 ("I get a kick out of figuring out why something isn't working.", "I", "positive"),
 ("Reading up on how things work is honestly fun for me.", "I", "positive"),
 ("I enjoy digging into data until I find the pattern.", "I", "positive"),
 ("Doing research all day sounds really boring to me.", "I", "negative"),
 ("I don't enjoy solving abstract problems, it just drains me.", "I", "negative"),
 ("I love designing posters and playing around with layouts.", "A", "positive"),
 ("Writing stories is something I'd happily do for hours.", "A", "positive"),
 ("I enjoy coming up with my own ideas instead of following a template.", "A", "positive"),
 ("Creative work stresses me out, I prefer clear instructions.", "A", "negative"),
 ("I'm not into anything artsy, drawing or music never appealed to me.", "A", "negative"),
 ("Helping my friends through a rough patch feels really meaningful.", "S", "positive"),
 ("I love tutoring juniors and seeing it finally click for them.", "S", "positive"),
 ("I enjoy volunteering and working with people face to face.", "S", "positive"),
 ("Taking care of other people's problems all day sounds exhausting.", "S", "negative"),
 ("I don't really enjoy teaching, I lose patience quickly.", "S", "negative"),
 ("I like pitching ideas and convincing people to get on board.", "E", "positive"),
 ("Leading a team and making the final call is exciting to me.", "E", "positive"),
 ("I enjoy negotiating, even just haggling at a flea market.", "E", "positive"),
 ("Selling things to people makes me really uncomfortable.", "E", "negative"),
 ("I'd hate to be the one in charge and responsible for everyone.", "E", "negative"),
 ("I find it satisfying to keep my spreadsheets neat and organised.", "C", "positive"),
 ("I enjoy following a clear process and getting the details right.", "C", "positive"),
 ("Sorting and tidying up records is weirdly relaxing for me.", "C", "positive"),
 ("Routine paperwork bores me to tears.", "C", "negative"),
 ("I can't stand checking numbers line by line.", "C", "negative"),
]
FAM = [  # reason = familiarity: stance comes from exposure, not stated enjoyment
 ("My dad runs a workshop so I've always been around engines.", "R"),
 ("I did engineering in poly, so hands-on stuff is what I know.", "R"),
 ("I've worked part-time in a warehouse for two years, so that's my experience.", "R"),
 ("I took the electrical course because my cousin did it too.", "R"),
 ("I grew up helping on my uncle's renovation jobs.", "R"),
 ("I'm a science student, so research is the obvious path for me.", "I"),
 ("I did a lab internship last summer, so I guess lab work.", "I"),
 ("Everyone in my family is a doctor or scientist, so that's what I'm used to.", "I"),
 ("I majored in data science because it seemed like the safe choice.", "I"),
 ("My CCA was the coding club, so programming is what I've done.", "I"),
 ("I studied design in poly so I've only really done creative work.", "A"),
 ("I've been in the school band since primary school.", "A"),
 ("My internship was at a media agency, so content is what I know.", "A"),
 ("I took art as an elective because my friends did.", "A"),
 ("My mum is a photographer so I kind of grew into it.", "A"),
 ("I've been a swim coach part-time, so teaching kids is what I'm used to.", "S"),
 ("My parents are both teachers, so education is familiar.", "S"),
 ("I did my internship in HR, so I know the people side.", "S"),
 ("I'm in social work because that was the course my grades got me into.", "S"),
 ("I've volunteered at the same eldercare centre for years.", "S"),
 ("I did the marketing track so that's what I know.", "E"),
 ("My family runs a business, so I've always helped with sales.", "E"),
 ("I was student council president, so leading things is what I've done.", "E"),
 ("I've done retail sales part-time since secondary school.", "E"),
 ("I'm in business school, so management seems like where I'll end up.", "E"),
 ("I'm studying accountancy, so I suppose I'll be an accountant.", "C"),
 ("My internship was in audit, so that's my experience so far.", "C"),
 ("I've done admin work at my mum's clinic for a few years.", "C"),
 ("I picked finance because my parents said it's stable.", "C"),
 ("I've only ever done data entry jobs, so that's what I know.", "C"),
]
NEU = [  # no stance on any interest, or off-topic; reason = neither
 "I'm in my final year and not really sure.",
 "I haven't thought about it much yet.",
 "Hmm, I don't know.",
 "Can you repeat the question?",
 "I'm graduating next May.",
 "I'm 22 and studying at SMU.",
 "Not sure, maybe?",
 "It depends, I guess.",
 "I had chicken rice for lunch.",
 "I'm just trying to figure things out.",
 "My friends are all applying for internships now.",
 "Honestly no idea.",
 "I'll think about it.",
 "That's a hard question.",
 "I just want something that pays the bills.",
 "I live in Tampines.",
 "Ok.",
 "I'm taking five modules this term.",
 "What do you mean by that?",
 "Both sound fine I guess.",
 "Either is okay with me.",
 "I don't have a strong preference.",
 "I'm still exploring my options.",
 "Hard to say right now.",
 "I've been really busy with exams lately.",
 "Let me think.",
 "I'm not sure what I want to do after graduating.",
 "Whatever works, really.",
 "My exchange semester is next term.",
 "I don't mind either way.",
]
seeds = []
for i, (t, ty, pol) in enumerate(ENJ, 1):
    seeds.append({"text": t, "slots": slots(**{ty: pol}), "reason": "enjoyment", "seed_id": f"ENJ_{i:03d}"})
for i, (t, ty) in enumerate(FAM, 1):
    seeds.append({"text": t, "slots": slots(**{ty: "positive"}), "reason": "familiarity", "seed_id": f"FAM_{i:03d}"})
for i, t in enumerate(NEU, 1):
    seeds.append({"text": t, "slots": slots(), "reason": "neither", "seed_id": f"NEU_{i:03d}"})
with open(f"{OUT}/synthetic_seeds.jsonl", "w") as f:
    for r in seeds: f.write(json.dumps(r) + "\n")

# question bank
CLARIFY = {
 "Realistic": [
  "Would you rather be hands-on with equipment and materials, or working through information at a desk?",
  "If you had to pick, would you rather fix a broken machine or write a report about why it broke?",
  "Does working outdoors or in a workshop appeal to you more than working in an office?"],
 "Investigative": [
  "Would you rather figure out why something happens, or just get it done and move on?",
  "Does spending a day analysing data to find a pattern sound good, or would you rather be doing something more active?",
  "If a problem has no obvious answer, do you want to dig into it, or hand it to someone else?"],
 "Artistic": [
  "Would you want room to make the work your own, or a clear brief to follow?",
  "Would you rather design how something looks, or make sure it works as specified?",
  "Does a job where you come up with new ideas every week appeal to you, or does that sound stressful?"],
 "Social": [
  "Would you rather spend your day helping or teaching people, or working mostly on your own tasks?",
  "If a colleague is struggling, would you want to be the one who sits down and helps them through it?",
  "Does a job where you talk to people all day sound energising or draining?"],
 "Enterprising": [
  "Would you rather be the one pitching the idea and persuading people, or the one doing the work behind it?",
  "Do you want to lead a team and make the final call, or would you rather someone else carries that?",
  "Does negotiating a deal or selling something sound exciting or uncomfortable?"],
 "Conventional": [
  "Would you rather follow a clear process and get the details right, or work things out as you go?",
  "Does keeping records, schedules and numbers in order sound satisfying or tedious?",
  "Would you prefer a job with a predictable routine, or one where every day is different?"],
}
ANSWERS = {
 "Realistic": [
  {"pos": "Hands-on, definitely. I like working with actual equipment.", "neg": "Desk, for sure. I'm not a hands-on person."},
  {"pos": "I'd rather fix the machine.", "neg": "I'd rather write the report, I'm not good with machines."},
  {"pos": "Yeah, a workshop or outdoors sounds better than sitting in an office.", "neg": "No, I'd rather be in an office."}],
 "Investigative": [
  {"pos": "I want to know why it happens.", "neg": "Just get it done, I don't need the theory."},
  {"pos": "That sounds good actually, I like finding patterns.", "neg": "That sounds boring, I'd want something more active."},
  {"pos": "I'd want to dig into it until I figure it out.", "neg": "I'd probably hand it to someone else."}],
 "Artistic": [
  {"pos": "Yeah, I like having room to make it my own.", "neg": "Not really, I'd rather follow a clear brief."},
  {"pos": "Designing how it looks, that's the fun part.", "neg": "Making sure it works, I don't care much about looks."},
  {"pos": "That appeals to me, I like coming up with ideas.", "neg": "That sounds stressful, I'd rather not have to be creative."}],
 "Social": [
  {"pos": "Helping people, I'd like that.", "neg": "Working on my own, I prefer that."},
  {"pos": "Yes, I'd want to help them through it.", "neg": "Not really, I'd rather focus on my own work."},
  {"pos": "Energising, I like talking to people.", "neg": "Draining, I need time away from people."}],
 "Enterprising": [
  {"pos": "Pitching and persuading, I enjoy that.", "neg": "Doing the work behind it, I don't like selling ideas."},
  {"pos": "I'd want to lead and make the call.", "neg": "I'd rather someone else carries that."},
  {"pos": "Exciting, I like negotiating.", "neg": "Uncomfortable, I don't like selling."}],
 "Conventional": [
  {"pos": "A clear process, I like getting the details right.", "neg": "Work it out as I go, processes feel restrictive."},
  {"pos": "Satisfying, I like things in order.", "neg": "Tedious, I'd hate that."},
  {"pos": "A predictable routine suits me.", "neg": "Every day different, routine bores me."}],
}
bank = {
 "OPENING": ["Hi! Tell me a bit about what you're studying and what kind of work you can see yourself enjoying."],
 "CLARIFY": CLARIFY,
 "CHALLENGE": [
  "Earlier you said \"{old_src}\", but just now \"{new_src}\". Which is closer to how you'd actually want to spend a week?",
  "I want to check something: you said \"{old_src}\" and also \"{new_src}\". Which one matters more to you?",
  "Those two sound a bit different: \"{old_src}\" and \"{new_src}\". If you could only have one, which would it be?",
  "Help me understand: \"{old_src}\" versus \"{new_src}\". Which feels more true for you?"],
 "PROBE": [
  "You said \"{new_src}\" — is that something you'd enjoy, or mainly what you've had experience with so far?",
  "When you say \"{new_src}\", is that because you like it, or because it's what you know?",
  "Imagine you'd never done it before. Would \"{new_src}\" still appeal to you?",
  "If you could pick freely, regardless of your background, would you still choose what you described in \"{new_src}\"?"],
 "ABSTAIN": [
  "No worries, let me put it another way.",
  "That's fair, it's a tricky one. Let me ask differently.",
  "Okay, let me try a more concrete question."],
 "RECOMMEND": [
  "Based on what you've told me, here are five occupations that fit, with the tasks that matched what you said:",
  "Here are five that fit you, and the parts of each job that match your answers:"],
 "ANSWERS": ANSWERS,
}
json.dump(bank, open(f"{OUT}/question_bank.json", "w"), indent=2)

# 60 NLI pairs
CON = [  # genuine contradictions on one slot
 ("I like bouncing ideas off people", "I get more done when nobody's around and I hate group work", "Social"),
 ("I love teaching my juniors", "I really don't enjoy explaining things to people", "Social"),
 ("Helping people is what motivates me", "I'd rather not deal with other people's problems", "Social"),
 ("I enjoy working with my hands", "I hate any kind of manual work", "Realistic"),
 ("I'd love a job outdoors", "I want to stay in an air-conditioned office all day", "Realistic"),
 ("Fixing machines is fun for me", "Machines and tools don't interest me at all", "Realistic"),
 ("I love digging into data to find patterns", "Analysing data is the most boring thing for me", "Investigative"),
 ("I want to understand why things work", "I don't care how things work, I just want to use them", "Investigative"),
 ("Research sounds exciting to me", "I'd find doing research all day really tedious", "Investigative"),
 ("I love designing things from scratch", "I don't enjoy creative work, I prefer clear instructions", "Artistic"),
 ("Writing is my favourite thing to do", "I really dislike writing anything", "Artistic"),
 ("I want freedom to do things my own way", "I'd rather be told exactly what to do", "Artistic"),
 ("I want to lead a team someday", "I never want to be in charge of other people", "Enterprising"),
 ("I enjoy pitching ideas to people", "Persuading people makes me really uncomfortable", "Enterprising"),
 ("Sales sounds exciting to me", "I would hate a job in sales", "Enterprising"),
 ("I like taking risks on new ventures", "I avoid risk and don't want to start anything of my own", "Enterprising"),
 ("I like keeping everything organised and tidy", "Organising files and records bores me to death", "Conventional"),
 ("I prefer a predictable routine", "I hate routine, I need every day to be different", "Conventional"),
 ("Checking details carefully is satisfying", "I can't stand going through details line by line", "Conventional"),
 ("I enjoy working with spreadsheets", "Spreadsheets are my least favourite thing", "Conventional"),
]
ELAB = [  # opposite signs on the slot, but compatible
 ("I like working with people", "I need quiet to focus when I'm writing", "Social"),
 ("I enjoy tutoring", "I don't like large crowds", "Social"),
 ("I like helping my friends", "I don't want to be a counsellor", "Social"),
 ("I enjoy building things", "I don't want to do heavy lifting every day", "Realistic"),
 ("I like fixing my bike", "I'm not interested in fixing cars", "Realistic"),
 ("I'd like some time outdoors", "I wouldn't want to be outside in the rain all day", "Realistic"),
 ("I like solving problems", "I don't enjoy pure maths proofs", "Investigative"),
 ("I enjoy reading about science", "I don't want to work in a wet lab", "Investigative"),
 ("I like analysing data", "Theoretical research isn't for me", "Investigative"),
 ("I enjoy design", "I'm not good at drawing by hand", "Artistic"),
 ("I like writing", "I don't enjoy poetry", "Artistic"),
 ("I like some creative freedom", "I don't want to be a full-time artist", "Artistic"),
 ("I like leading small projects", "I don't want to manage a huge department", "Enterprising"),
 ("I enjoy persuading people", "I don't like cold-calling strangers", "Enterprising"),
 ("I'd like to start something someday", "I wouldn't want to gamble all my savings", "Enterprising"),
 ("I like being organised", "I don't want a job that is only data entry", "Conventional"),
 ("I enjoy working with numbers", "I don't like doing taxes", "Conventional"),
 ("I like a clear process", "I don't want to be micromanaged", "Conventional"),
 ("I like keeping records tidy", "Filing paper documents is not for me", "Conventional"),
 ("I like routine in my mornings", "I don't want every task at work to be identical", "Conventional"),
]
UNREL = [
 ("I like working with people", "I'm in my final year at SMU", "Social"),
 ("I enjoy teaching", "I had chicken rice for lunch", "Social"),
 ("Helping others motivates me", "My exchange semester is next term", "Social"),
 ("I enjoy working with my hands", "I'm taking five modules this term", "Realistic"),
 ("I'd love a job outdoors", "My friends are all applying for internships", "Realistic"),
 ("Fixing machines is fun", "I live in Tampines", "Realistic"),
 ("I love analysing data", "I'm graduating next May", "Investigative"),
 ("Research excites me", "I've been busy with exams lately", "Investigative"),
 ("I want to understand how things work", "My sister just started university", "Investigative"),
 ("I love designing things", "I take the MRT to school every day", "Artistic"),
 ("Writing is my favourite thing", "I'm 22 years old", "Artistic"),
 ("I want creative freedom", "It's been raining a lot this week", "Artistic"),
 ("I want to lead a team", "I usually study in the library", "Enterprising"),
 ("I enjoy pitching ideas", "I just got back from holiday", "Enterprising"),
 ("Sales sounds exciting", "My laptop is getting old", "Enterprising"),
 ("I like keeping things organised", "I play badminton on weekends", "Conventional"),
 ("I prefer a predictable routine", "My family has a dog", "Conventional"),
 ("Checking details is satisfying", "I'm not sure what to eat tonight", "Conventional"),
 ("I enjoy spreadsheets", "My course has a group project due soon", "Conventional"),
 ("I like a clear process", "I'm from Singapore", "Conventional"),
]
pairs = []
for lab, arr in (("contradiction", CON), ("elaboration", ELAB), ("unrelated", UNREL)):
    idx = list(range(20)); random.shuffle(idx); test = set(idx[:7] if lab != "unrelated" else idx[:6])
    for i, (a, b, s) in enumerate(arr):
        pairs.append({"id": f"{lab[:3].upper()}_{i+1:02d}", "a": a, "b": b, "slot": s, "label": lab,
                      "split": "test" if i in test else "dev"})
with open(f"{OUT}/nli_pairs.jsonl", "w") as f:
    for p in pairs: f.write(json.dumps(p) + "\n")
# unlabelled copy for the blind human sort
blind = [{"id": p["id"], "a": p["a"], "b": p["b"]} for p in pairs]; random.shuffle(blind)
with open(f"{OUT}/nli_pairs_blind.jsonl", "w") as f:
    for p in blind: f.write(json.dumps(p) + "\n")

n_bank = 1 + 18 + 36 + 4 + 4 + 3 + 2
print("seeds", len(seeds), "| bank sentences", n_bank, "| nli", len(pairs),
      "dev", sum(p["split"] == "dev" for p in pairs), "test", sum(p["split"] == "test" for p in pairs))
