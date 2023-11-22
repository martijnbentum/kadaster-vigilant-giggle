import json
import Levenshtein 


class Kraken:
    def __init__(self, filename):
        self.filename = filename
        self.name = filename.split('/')[-1].split('.')[0]
        self.json = json.load(open(filename))
        self._make_words()

    def _make_words(self):
        self.words = []
        for index, line in enumerate(self.json):
            word = Word(index,line, self)
            self.words.append(word)

    def find_best_word_match(self, s):
        smallest = 10**6
        best_match= None
        self.perfect_matches = []
        for word in self.words:
            distance = word.levenshtein(s)
            if distance < smallest: 
                best_match= word
                smallest = distance
            if distance == 0: self.perfect_matches.append(word)
        if len(self.perfect_matches) > 0: 
            print('found', len(self.perfect_matches), 'perfect match(es)')
            print('check attribute .perfect_matches')
        return best_match, distance
        
    def find_partial_match(self,s):
        output = []
        for word in self.words:
            if s in word.text: output.append(word)
        return output



class Word:
    def __init__(self,index, json_line, parent):
        self.index = index
        self.json = json_line
        self.parent = parent
        self.location = json_line['location']
        self.text = json_line['text']

    def __repr__(self):
        return 'word: '+self.text

    def _set_info(self):
        x1,y1,x2,y2 = self.location
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.width = x2 - x1
        self.height = y2 - y1

    def levenshtein(self, other):
        if type(other) == str:
            return Levenshtein.distance(self.text, other)
        return Levenshtein.distance(self.text, other.text)

    @property
    def coordinates(self):
        x1,y1,x2,y2 = self.location
        return x1,y1,x2,y2

    @property
    def center(self):
        x1,y1,x2,y2 = self.location
        return (x1+x2)/2, (y1+y2)/2

    def distance(self, other):
        x1, y1= self.center
        x2, y2= other.center
        return ((x1-x2)**2 + (y1-y2)**2)**0.5
        

