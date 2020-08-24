from abc import ABCMeta, abstractmethod
from random import randint, randrange


'''所有角色都有的属性和方法'''
class Role():
    def __init__(self,name,hp,mp):
        self.name=name
        self.hp=hp
        self.mp=mp

    @property
    def name(self):
        return self.name

    @property
    def hp(self):
        return self.hp

    @property
    def mp(self):
        return self.mp

    @property
    def alive(self):
        return self.hp > 0

    @abstractmethod
    def attack(self,be_attacker):
        pass


class Soldier(Role):

    def __init__(self,name,hp,mp):
        super().__init__(name,hp,mp)

    def attack(self,be_attacker):
        be_attacker.hp-=randint(100,500)

    def huge_attack(self,be_attacker):
        lowest_injury=200
        if self.mp>50:
            self.mp-=50
            injury=self.hp*9//10
            injury= injury if injury>lowest_injury else lowest_injury
            be_attacker.hp-=injury
            return True
        else:
            return False

    def resistance(self):
        return True

    def magic_attack(self,be_attackers):
        if self.mp>30:
            self.mp-=30
            for role in be_attackers:
                injury=randint(100,200)
                if role.alive:
                    role.hp-=injury
            return True
        else:
            return False


class Monster(Role):
    def __init__(self,name,hp,mp):
        super().__init__(name,hp,mp)

    def attack(self,be_attacker):
        be_attacker.hp-=randint(100,500)

    def magic_attack(self,be_attackers):
        if self.mp>30:
            self.mp-=30
            for role in be_attackers:
                injury=randint(100,200)
                if role.alive:
                    role.hp-=injury
            return True
        else:
            return False

    def resistance(self):
        return True


class SelectRole():
    def __init__(self,roles):
        self.roles=roles

    def select_role(self):
        while True:
            role_len=len(self.roles)
            index = randrange(role_len)
            role=self.roles[index]
            if role.alive:
                return role


class LiveStatus(Role):
    def is_any_live(self,roles):
        for role in roles:
            if role.alive:
                return True
            else:
                return False


def main():
    soldier=Soldier('zs',10000,500)
    magicer=Soldier('fs',2000,1000)
    m_1 = Monster('ms_1',5000,100)
    m_2 = Monster('ms_2',3000,200)
    m_3 = Monster('ms_2', 3000, 200)

    heros=[soldier,magicer]
    mosters=[m_1,m_2,m_3]

    fight_round=1

    while LiveStatus.is_any_live(heros) and LiveStatus.is_any_live(mosters):
        print('========the%02dround========' % fight_round)
        print('heros attack')
        skill_value=randint(1,100)
        for hero in heros:
            if hero.alive:
                pass




