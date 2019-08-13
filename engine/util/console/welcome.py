import random

from engine.util.console.print_colors import color


welcome = ["""
 _______________________________________
( Let's run some experiments, shall we? )
 ---------------------------------------
        o   ^__^
         o  (oo)\\_______
            (__)\\       )\\/\\
                ||----w |
                ||     ||

""",
"""
 _____________________________________________
/ Consider some GPUs to make your experiment  \\
\\ faster.                                     /
 ---------------------------------------------
    \\                                  ___-------___
     \\                             _-~~             ~~-_
      \\                         _-~                    /~-_
             /^\\__/^\\         /~  \\                   /    \\
           /|  O|| O|        /      \\_______________/        \\
          | |___||__|      /       /                \\          \\
          |          \\    /      /                    \\          \\
          |   (_______) /______/                        \\_________ \\
          |         / /         \\                       /            \\
           \\         \\^\\          \\                   /               \\     /
             \\         ||           \\_______________/      _-_       //\\__//
               \\       ||------_-~~-_ ------------- \\ --/~   ~\\    || __/
                 ~-----||====/~     |==================|       |/~~~~~
                  (_(__/  ./     /                    \\_\\      \\.
                         (_(___/                         \\_____)_)
""",
"""
 _______________________________________________________________
( I have been waiting for you. I am fuelled by neural networks. )
 ---------------------------------------------------------------
   o
    o______    ________
    /      \\_/         |
   |                 | |
   |                 | |
   |    ###\\  /###   | |
   |     0  \\/  0    | |
  /|                 | |
 / |        <        |\\ \\
| /|                 | | |
| ||    \\_______/    | | |
| ||                 |/ /
/|||                /|||
   |-----------------|
        | |    | |
        ***    ***
       /___\\  /___\\
""",
"""
 ____________
( << WAIT >> )
 ------------
     o
      o
        ,__, |    | 
        (oo)\\|    |___
        (__)\\|    |   )\\_
             |    |_w |  \\
             |    |  ||   *

             Cower....
""",
"""
  _________________________________
/ Hello! Hope your experiment will \\
\\ obtain nice results.             /
 ----------------------------------
   \\
    \\
               |    .
           .   |L  /|
       _ . |\ _| \--+._/| .
      / ||\| Y J  )   / |/| ./
     J  |)'( |        ` F`.'/
   -<|  F         __     .-<
     | /       .-'. `.  /-. L___
     J \      <    \  | | O\|.-'
   _J \  .-    \/ O | | \  |F
  '-F  -<_.     \   .-'  `-' L__
 __J  _   _.     >-'  )._.   |-'
 `-|.'   /_.           \_|   F
   /.-   .                _.<
  /'    /.'             .'  `\\
   /L  /'   |/      _.-'-\\
  /'J       ___.---'\|
    |\  .--' V  | `. `
    |/`. `-.     `._)
       / .-.\\
 VK    \ (  `\\
        `.\\


""",
"""
 _____________________________________
/ I am so much more smarter than your \\
\\ model with my huge brain!           /
 -------------------------------------
   \\         __------~~-,
    \\      ,'            ,
          /               \\
         /                :
        |                  '
        |                  |
        |                  |
         |   _--           |
         _| =-.     .-.   ||
         o|/o/       _.   |
         /  ~          \\ |
       (____@)  ___~    |
          |_===~~~.`    |
       _______.--~     |
       \________       |
                \\      |
              __/-___-- -__
             /            _ \\
""",
"""
 __________________________
( Man, you are killing me! )
 --------------------------
             o
              o  (__)      
               o /oo|  
                (_"_)*+++++++++*
                   //I#\\\\\\\\\\\\\\\\I\\
                   I[I|I|||||I I `
                   I`I'///'' I I
                   I I       I I
                   ~ ~       ~ ~
                     Scowleton
""",
"""
 ________________________________________
( Anything is possible, unless it's not. )
 ----------------------------------------
  o
   o          .
       ___   //
     {~._.~}// 
      ( Y )K/  
     ()~*~()   
     (_)-(_)   
     Luke    
     Skywalker
     koala  
""",
"""
 ______________________________________
( For fuck sake, leave me alone! I'm  )
( doing some experiments right now... )
 --------------------------------------
      o                _
       o              (_)
        o   ^__^       / \\
         o  (oo)\\_____/_\\ \\
            (__)\\       ) /
                ||----w ((
                ||     ||>> 
""",
"""
 ______________________________________
( Please god! Make the biases within  )
( the data go away :s...              )
 --------------------------------------
             o
              o
                  .=\"=.
                _/.-.-.\\_     _
               ( ( o o ) )    ))
                |/  \"  \\|    //
                 \\'---'/    //
                 /`\"\"\"`\\\\  ((
                / /_,_\\ \\\\  \\\\
                \\_\\_'__/  \\  ))
                /`  /`~\\   |//
               /   /    \\  /
          ,--`,--'\\/\\    /
          \'-- \"--'  '--'
""",
"""
 ______________________________________
( Get the fuck out of my ass and let's )
( start some experiments               )
 --------------------------------------
    o
     o
    ^__^         /
    (oo)\\_______/  _________
    (__)\\       )=(  ____|_ \\_____
        ||----w |  \\ \\     \\_____ |
        ||     ||   ||           ||
""", """
 ______________________________________
( I am eating logistic regression for  )
( breakfast... Consider adding more    )
( layers...                            )
 --------------------------------------
                       o                    ^    /^
                        o                  / \\  // \\
                         o   |\\___/|      /   \\//  .\\
                          o  /O  O  \\__  /    //  | \\ \\           *---*
                            /     /  \\/_/    //   |  \\  \\         \\   |
                            @___@`    \\/_   //    |   \\   \\        \\/\\ \\
                           0/0/|       \\/_ //     |    \\    \\         \\ \\
                       0/0/0/0/|        \\///      |     \\     \\       |  |
                    0/0/0/0/0/_|_ /   (  //       |      \\     _\\     |  /
                 0/0/0/0/0/0/`/,_ _ _/  ) ; -.    |    _ _\\.-~       /   /
                             ,-}        _      *-.|.-~-.           .~    ~
            \\     \\__/        `/\\      /                 ~-. _ .-~      /
             \\____(oo)           *.   }            {                   /
             (    (--)          .----~-.\\        \\-`                 .~
             //__\\\\  \\__ Ack!   ///.----..<        \\             _ -~
            //    \\\\               ///-._ _ _ _ _ _ _{^ - - - - ~

""",
"""
 ______________________________________
( Let's kill some babies... I mean..   )
( Let's run some experiments!          )
 --------------------------------------
   o         ,        ,
    o       /(        )`
     o      \\ \\___   / |
            /- _  `-/  '
           (/\\/ \\ \\   /\\
           / /   | `    \\
           O O   ) /    |
           `-^--'`<     '
          (_.)  _  )   /
           `.___/`    /
             `-----' /
<----.     __ / __   \\
<----|====O)))==) \\) /====
<----'    `--' `.__,' \\
             |        |
              \\       /
        ______( (_  / \\______
      ,'  ,-----'   |        \\
      `--{__________)        \\/
""",
"""
 ________________
( So...          )
( Where I live ? )
 ----------------
 o       
  o        
   o   
    o  

      ccee88oo
  C8O8O8Q8PoOb o8oo
 dOB69QO8PdUOpugoO9bD
CgggbU8OU qOp qOdoUOdcb
    6OuU  /p u gcoUodpP
      \\\\\\//  /douUP
        \\\\\\////
         |||/\\
         |||\/
         |||||
   .....//||||\\....
"""
]

rick_hair = color.OVER_TURQUOISE + color.TURQUOISE + '█' + color.END
rick_skin = color.OVER_GRAY + color.GRAY + '█' + color.END
rick_eyebrow = color.OVER_DARKCYAN + color.DARKCYAN + '█' + color.END
rick_eye = color.OVER_LIGHTGRAY + color.LIGHTGRAY + '█' + color.END
rick_c = color.OVER_BLACK + color.BLACK + '█' + color.END
rick_mouth = color.OVER_R_BROWN + color.R_BROWN + '█' + color.END

morty_hair = color.OVER_BROWN + color.BROWN + '█' + color.END
morty_skin = color.OVER_PALE_PINK + color.PALE_PINK + '█' + color.END
goodbye = ["""
_________________________________________
( What is my purpose? You pass butter..   )
( Hopefully your experiments went well... )
 -----------------------------------------
     o 
      o       """ + rick_hair * 2 + """    """ + rick_hair * 2 + """            
       o    """ + rick_hair * 6 + """  """ + rick_hair * 4 + """
            """ + rick_hair * 12 + """
      """ + rick_hair * 8 + rick_skin * 8 + rick_hair * 6 + """
        """ + rick_hair * 4 + rick_skin * 12 + rick_hair * 2 + """
          """ + rick_hair * 2 + rick_skin * 2 + rick_eyebrow * 10 + """
        """ + rick_hair * 4 + rick_skin * 2 + rick_eye * 4 + rick_skin * 2 + rick_eye * 4 + rick_hair * 2 + """
      """ + rick_hair * 6 + rick_skin * 2 + rick_eye * 2 + rick_c * 2 + rick_skin * 2 + rick_eye * 2 + rick_c * 2 \
          + rick_hair * 4 + """
        """ + rick_hair * 2 + rick_skin * 14 + rick_hair * 2 + """
          """ + rick_hair * 2 + rick_skin * 12 + """
        """ + rick_hair * 4 + rick_skin * 4 + rick_mouth * 6 + rick_skin * 2 + """
              """ + rick_skin * 8 +  """
                """ + rick_skin * 4 + """
""", """
 _________________________________________
( One does not simply walk into success...)
( But hopefully, you did!                 )
 -----------------------------------------
  o
   o
                     _____
                   .'* *.'
               ___/_*_(_
              / _______ \\
             _\\_)/___\\(_/_
            / _((\\- -/))_ \\
            \\ \\())(-)(()/ /
             ' \\(((()))/ '
            / ' \\)).))\\ ' \\
           / _ \\ - | - /_  \\
          (   ( .;''';. .'  )
          _\\"__ /    )\\ __"/_
            \\/  \\   ' /  \\/
             .'  '...' '  )
              / /  |   \\  \\
             / .   .    .  \\
            /   .      .    \\
           /   /   |    \\    \\
         .'   /    b     '.   '.
     _.-'    /     Bb      '-.  '-_
 _.-'       |      BBb        '-.  '-.
(________mrf____.dBBBb._________)____)
""", """
 __________________________________
( I am burning to see your results )
 ----------------------------------
  o            .    .     .   
   o      .  . .     `  ,     
    o    .; .  : .' :  :  : . 
     o   i..`: i` i.i.,i  i . 
      o   `,--.|i |i|ii|ii|i: 
           UooU\\.'@@@@@@`.||' 
           \\__/(@@@@@@@@@@)'  
                (@@@@@@@@)    
                `YY~~~~YY'    
                 ||    ||    
""",
"""
 _________________________________
( I had fun once.... But then     )
( I saw your results :'( ..       )
 ----------------------------------
  o       
   o        
    o   
     o  
      
       ﾊ _ ﾊ
       ಠ X ಠ
"""]


def print_welcome_message():
    color_start = random.choice([color.GREEN, color.CYAN, color.BLUE, color.YELLOW])
    msg = random.choice(welcome)
    print(color_start + msg + color.END)


def print_goodbye():
    color_start = random.choice([color.GREEN, color.CYAN, color.BLUE, color.YELLOW])
    print(color_start + random.choice(goodbye) + color.END)
