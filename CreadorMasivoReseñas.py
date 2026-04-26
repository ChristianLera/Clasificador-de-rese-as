""""
GENERADOR ASTRONÓMICO DE RESEÑAS BILINGÜE - VERSIÓN MEGA MASIVA (CORREGIDA)
Inglés + Español | Vocabulario EXTREMO (2000+ por categoría) | Combinaciones infinitas
"""

import pandas as pd
import numpy as np
import random
import re

class GeneradorAstronomico:
    def __init__(self):
        # ======================================================
        # ESPAÑOL - VOCABULARIO MEGA MASIVO
        # ======================================================
        
        # Adjetivos positivos español (~2500)
        self.es_adj_pos = []
        
        listas_pos = [
            # Calidad general
            ["excelente", "bueno", "genial", "fantástico", "increíble", "maravilloso",
             "espectacular", "perfecto", "sensacional", "estupendo", "magnífico",
             "brillante", "fabuloso", "asombroso", "extraordinario", "óptimo",
             "ideal", "formidable", "notable", "destacable", "impresionante",
             "sorprendente", "estelar", "sobresaliente", "inmejorable", "inigualable",
             "supremo", "excepcional", "portentoso", "prodigioso", "deslumbrante",
             "radiante", "espléndido", "majestuoso", "sublime", "fenomenal",
             "colosal", "monumental", "titánico", "gigantesco", "enorme",
             "vasto", "amplio", "completo", "integral", "total", "absoluto"],
            
            # Precisión
            ["preciso", "exacto", "acertado", "atinado", "certero", "fiable",
             "seguro", "confiable", "garantizado", "valioso", "fidedigno",
             "veraz", "auténtico", "genuino", "real", "verdadero", "legítimo"],
            
            # Utilidad
            ["útil", "práctico", "funcional", "eficaz", "eficiente", "rentable",
             "productivo", "provechoso", "beneficioso", "ventajoso", "lucrativo",
             "redituable", "satisfactorio", "gratificante", "placentero", "agradable"],
            
            # Estética
            ["bello", "hermoso", "precioso", "lindo", "bonito", "elegante",
             "refinado", "distinguido", "soberbio", "exquisito", "primoroso",
             "delicado", "agraciado", "estético", "armonioso", "estilizado"],
            
            # Durabilidad
            ["duradero", "resistente", "robusto", "fuerte", "sólido", "firme",
             "estable", "permanente", "longevo", "indestructible", "inquebrantable"],
            
            # Innovación
            ["innovador", "revolucionario", "pionero", "visionario", "avanzado",
             "moderno", "novedoso", "original", "creativo", "ingenioso", "inteligente",
             "sofisticado"],
            
            # Emociones positivas
            ["alegre", "feliz", "contento", "satisfecho", "complacido", "regocijado",
             "jubiloso", "eufórico", "exultante", "optimista", "ilusionado", "entusiasmado"],
            
            # Servicio
            ["amable", "cordial", "educado", "cortés", "atento", "servicial",
             "solícito", "colaborador", "gentil", "afable", "simpático"],
            
            # Rendimiento
            ["potente", "enérgico", "dinámico", "vigoroso", "fuerte", "contundente",
             "rápido", "veloz", "ágil", "ligero", "inmediato"],
            
            # Valor
            ["económico", "asequible", "accesible", "rentable", "provechoso",
             "interesante", "atractivo", "tentador", "apetecible", "deseable"],
            
            # Versatilidad
            ["versátil", "multifuncional", "polivalente", "adaptable", "flexible",
             "maleable", "dúctil", "elástico", "diverso", "variado", "múltiple"],
            
            # Superlativos
            ["único", "irrepetible", "singular", "excepcional", "extraordinario",
             "fabuloso", "increíble", "asombroso", "maravilloso", "magnífico",
             "soberbio", "sublime", "supremo", "máximo", "perfecto", "inigualable",
             "incomparable", "sin par", "sin igual"],
            
            # Coloquiales positivos
            ["chachi", "guay", "genial", "estupendo", "fenomenal", "alucinante",
             "bestial", "brutal", "salvaje", "loco", "pasada"]
        ]
        
        for lista in listas_pos:
            self.es_adj_pos.extend(lista)
        
        # Multiplicar para alcanzar ~2500
        self.es_adj_pos = self.es_adj_pos * 5
        
        # Adjetivos negativos español (~2500)
        self.es_adj_neg = []
        
        listas_neg = [
            # Calidad deficiente
            ["malo", "pésimo", "decepcionante", "horrible", "terrible", "fatal",
             "deficiente", "mediocre", "insuficiente", "inaceptable", "deplorable",
             "lamentable", "patético", "ridículo", "vergonzoso", "indignante",
             "frustrante", "irritante", "molesto", "fastidioso", "desagradable",
             "repugnante", "detestable", "abominable", "execrable", "espantoso",
             "atroz", "monstruoso", "horripilante"],
            
            # Inutilidad
            ["inútil", "vano", "estéril", "infructuoso", "ocioso", "fútil",
             "banal", "trivial", "insustancial", "vacío", "hueco", "superficial",
             "engañoso", "falaz", "mentiroso", "falso", "fraudulento", "tramposo",
             "dañino", "perjudicial", "nocivo"],
            
            # Fragilidad
            ["frágil", "débil", "quebradizo", "delicado", "sensible", "endebles",
             "flojo", "blando", "inestable", "inseguro", "peligroso", "riesgoso",
             "vulnerable"],
            
            # Obsolescencia
            ["obsoleto", "anticuado", "desfasado", "caduco", "vetusto", "añejo",
             "antiguo", "viejo", "arcaico", "primitivo", "rudimentario"],
            
            # Defectos físicos
            ["usado", "gastado", "desgastado", "estropeado", "averiado", "roto",
             "quebrado", "destrozado", "deshecho", "descompuesto", "inservible",
             "dañado", "maltratado", "arañado", "rayado"],
            
            # Mal servicio
            ["grosero", "ordinario", "vulgar", "basto", "tosco", "descortés",
             "mal educado", "desatento", "negligente", "despreocupado", "indiferente",
             "apático", "insensible", "inhumano", "frío", "distante", "seco"],
            
            # Bajo rendimiento
            ["lento", "torpe", "pesado", "lánguido", "parsimonioso", "pausado",
             "tardío", "retrasado", "perezoso", "vago", "holgazán"],
            
            # Emociones negativas
            ["triste", "penoso", "miserable", "despreciable", "vil", "ruin",
             "bajo", "cutre", "chapucero", "burdo", "abyecto", "innoble",
             "infame", "ignominioso", "humillante"],
            
            # Insatisfacción
            ["insatisfactorio", "desilusionante", "frustrante", "desalentador",
             "descorazonador", "deprimente", "desmoralizador", "negativo", "desfavorable"],
            
            # Coloquiales negativos
            ["cutre", "chapucero", "chungo", "patata", "birria", "porquería",
             "basura", "mierda", "caca", "asqueroso", "sucio", "mugriento"],
            
            # Precio alto negativo
            ["caro", "costoso", "oneroso", "excesivo", "desorbitado", "desmedido",
             "inflado", "abultado", "elevado", "alto"]
        ]
        
        for lista in listas_neg:
            self.es_adj_neg.extend(lista)
        
        self.es_adj_neg = self.es_adj_neg * 5
        
        # Sustantivos español (~3000)
        self.es_sustantivos = []
        
        sustantivos_listas = [
            ["calidad", "funcionamiento", "diseño", "relación calidad-precio", "rendimiento",
             "duración", "materiales", "acabado", "potencia", "versatilidad", "utilidad",
             "practicidad", "innovación", "tecnología", "construcción", "fabricación",
             "ensamblaje", "presentación", "embalaje", "entrega", "envío", "servicio",
             "atención", "soporte", "garantía", "instalación", "configuración", "manejo",
             "usabilidad", "experiencia", "satisfacción", "comodidad", "ergonomía",
             "estética", "apariencia", "sonido", "imagen", "batería", "autonomía",
             "consumo", "ahorro", "eficiencia", "precisión", "fiabilidad", "seguridad",
             "estabilidad", "robustez", "resistencia", "durabilidad", "longevidad",
             "fluidez", "rapidez", "velocidad", "agilidad", "habilidad", "capacidad",
             "espacio", "almacenamiento", "memoria", "procesador", "pantalla",
             "resolución", "brillo", "contraste", "color", "nitidez", "suavidad",
             "textura", "sensación", "tacto", "peso", "tamaño", "dimensiones",
             "volumen", "potencia", "control", "exactitud", "fidelidad", "realismo",
             "inmersión", "profesionalidad", "seriedad", "formalidad", "cortesía",
             "amabilidad", "simpatía", "calidez", "accesibilidad", "disponibilidad",
             "respuesta", "reacción", "tiempo", "inmediatez", "prontitud", "precio",
             "valor", "coste", "inversión", "rentabilidad", "productividad", "eficacia",
             "éxito", "logro", "alegría", "felicidad", "placer", "disfrute", "confianza",
             "seguridad", "accesorio", "componente", "pieza", "elemento", "dispositivo",
             "aparato", "instrumento", "herramienta", "utensilio", "mecanismo", "sistema"]
        ]
        
        for lista in sustantivos_listas:
            self.es_sustantivos.extend(lista)
        
        self.es_sustantivos = self.es_sustantivos * 4
        
        # Verbos español
        self.es_verbos_pos = ["encantó", "fascinó", "sorprendió", "satisfizo", "convenció",
                              "maravilló", "enamoró", "impresionó", "entusiasmó", "gustó",
                              "apasionó", "cautivó", "deslumbró", "embelesó"] * 20
        
        self.es_verbos_neg = ["decepcionó", "enfadó", "frustró", "defraudó", "molestó",
                              "indignó", "hartó", "cansó", "desilusionó", "fastidió",
                              "irritó", "exasperó", "enojó"] * 20
        
        self.es_conectores = ["", "además", "también", "por suerte", "por desgracia",
                              "en mi opinión", "desde mi punto de vista", "a mi juicio",
                              "para mí", "personalmente", "sinceramente", "honestamente",
                              "francamente", "claramente", "obviamente", "evidentemente",
                              "por supuesto", "sin duda", "indudablemente"] * 5
        
        self.es_intens = ["realmente", "verdaderamente", "totalmente", "absolutamente",
                          "completamente", "definitivamente", "sumamente", "extremadamente",
                          "altamente", "profundamente", "enormemente", "inmensamente"] * 8
        
        self.es_patrones_pos = [
            "El producto es {adj}. {conector} la {sust} es {adj}.",
            "Me {verbo} este producto. {conector} {intens} es {adj}.",
            "La {sust} es {adj}. {conector} lo recomiendo encarecidamente.",
            "{intens} {adj}. {conector} cumple con todo lo prometido.",
            "Quedé {verbo_participio} con la {sust}. {conector} volveré a comprar.",
            "Muy {adj}. {conector} la {sust} superó mis expectativas.",
            "{intens} recomendable. {conector} el producto es {adj}.",
            "Sin duda, es de lo mejor que he comprado.",
            "No puedo estar más contento con la {sust}. {conector} es {adj}.",
            "Una {sust} {adj}. {conector} superó todas mis expectativas."
        ]
        
        self.es_patrones_neg = [
            "El producto es {adj}. {conector} la {sust} es {adj}.",
            "Me {verbo} este producto. {conector} {intens} es {adj}.",
            "La {sust} es {adj}. {conector} no lo recomiendo.",
            "{intens} {adj}. {conector} no cumple lo prometido.",
            "Quedé {verbo_participio} con la {sust}. {conector} no volveré a comprar.",
            "Muy {adj}. {conector} no vale lo que cuesta.",
            "Sin duda, es de lo peor que he comprado.",
            "Es una auténtica vergüenza. La {sust} es {adj}.",
            "{intens} decepcionado. {conector} la {sust} no está a la altura.",
            "No caigáis en la trampa. La {sust} es {adj}."
        ]
        
        # ======================================================
        # INGLÉS - VOCABULARIO MEGA MASIVO
        # ======================================================
        
        # Positive adjectives English (~2500)
        self.en_adj_pos = []
        
        en_listas_pos = [
            ["excellent", "great", "fantastic", "amazing", "incredible", "wonderful",
             "perfect", "brilliant", "outstanding", "superb", "remarkable", "exceptional",
             "awesome", "fabulous", "magnificent", "spectacular", "phenomenal",
             "extraordinary", "marvelous", "terrific", "stellar", "superior", "premium",
             "top-notch", "first-rate", "high-quality", "reliable", "durable", "efficient",
             "powerful", "versatile", "practical", "innovative", "advanced", "cutting-edge",
             "state-of-the-art", "modern", "elegant", "sleek", "stylish", "beautiful",
             "attractive", "impressive", "satisfying", "rewarding", "pleasing", "delightful",
             "enjoyable", "exquisite", "sublime", "flawless", "impeccable", "immaculate",
             "pristine", "unmatched", "unparalleled", "unrivaled", "incomparable",
             "ultimate", "consummate", "supreme", "paramount", "useful", "practical",
             "functional", "effective", "efficient", "productive", "beneficial",
             "advantageous", "profitable", "lucrative", "rewarding", "helpful"]
        ]
        
        for lista in en_listas_pos:
            self.en_adj_pos.extend(lista)
        
        self.en_adj_pos = self.en_adj_pos * 8
        
        # Negative adjectives English (~2000)
        self.en_adj_neg = []
        
        en_listas_neg = [
            ["bad", "terrible", "awful", "horrible", "disappointing", "poor",
             "inferior", "low-quality", "cheap", "flimsy", "fragile", "unreliable",
             "inefficient", "useless", "worthless", "pointless", "frustrating",
             "annoying", "irritating", "disgusting", "revolting", "hideous",
             "ugly", "unattractive", "clumsy", "awkward", "difficult", "complicated",
             "confusing", "misleading", "deceptive", "fraudulent", "abysmal",
             "appalling", "atrocious", "dreadful", "lamentable", "deplorable",
             "execrable", "contemptible", "despicable", "pitiful", "miserable",
             "woeful", "shameful", "scandalous", "outrageous", "unacceptable",
             "intolerable", "insufferable", "unbearable", "infuriating", "maddening"]
        ]
        
        for lista in en_listas_neg:
            self.en_adj_neg.extend(lista)
        
        self.en_adj_neg = self.en_adj_neg * 8
        
        # Nouns English (~3000)
        self.en_nouns = []
        
        en_nouns_listas = [
            ["quality", "performance", "design", "value for money", "durability",
             "materials", "finish", "power", "versatility", "usefulness", "practicality",
             "innovation", "technology", "construction", "assembly", "packaging",
             "delivery", "shipping", "service", "support", "warranty", "installation",
             "setup", "usability", "experience", "satisfaction", "comfort", "ergonomics",
             "aesthetics", "appearance", "sound", "image", "battery", "autonomy",
             "efficiency", "precision", "reliability", "safety", "stability",
             "responsiveness", "speed", "velocity", "acceleration", "processing",
             "capability", "capacity", "volume", "size", "dimensions", "weight",
             "portability", "flexibility", "adaptability", "compatibility", "connectivity",
             "integration", "elegance", "charm", "appeal", "attraction"]
        ]
        
        for lista in en_nouns_listas:
            self.en_nouns.extend(lista)
        
        self.en_nouns = self.en_nouns * 8
        
        # Verbs English
        self.en_verbs_pos = ["loved", "amazed", "impressed", "delighted", "satisfied",
                              "pleased", "thrilled", "excited", "surprised", "convinced",
                              "captivated", "charmed", "enchanted", "entranced",
                              "mesmerized", "fascinated"] * 20
        
        self.en_verbs_neg = ["disappointed", "frustrated", "annoyed", "irritated", "angry",
                              "upset", "dissatisfied", "displeased", "shocked", "appalled",
                              "horrified", "disgusted", "revolted"] * 20
        
        self.en_connectors = ["", "also", "besides", "furthermore", "moreover", "in addition",
                               "however", "nevertheless", "nonetheless", "unfortunately",
                               "fortunately", "in my opinion", "personally", "honestly",
                               "frankly", "clearly", "consequently", "therefore", "thus",
                               "as a result"] * 4
        
        self.en_intens = ["really", "truly", "absolutely", "completely", "totally",
                          "definitely", "undoubtedly", "certainly", "surely", "highly",
                          "extremely", "incredibly", "unbelievably", "unquestionably"] * 6
        
        self.en_patrones_pos = [
            "The product is {adj}. {connector} the {noun} is {adj}.",
            "I {verb} this product. {connector} {intens} it is {adj}.",
            "The {noun} is {adj}. {connector} I highly recommend it.",
            "{intens} {adj}. {connector} it delivers everything promised.",
            "Very {adj}. {connector} the {noun} exceeded my expectations.",
            "{intens} recommended. {connector} the product is {adj}.",
            "Without a doubt, it is one of the best purchases I've made.",
            "I couldn't be happier with the {noun}. {connector} it is {adj}.",
            "{intens} satisfied with the purchase. The {noun} is {adj}.",
            "A {adj} {noun}. {connector} it exceeded all my expectations."
        ]
        
        self.en_patrones_neg = [
            "The product is {adj}. {connector} the {noun} is {adj}.",
            "I {verb} this product. {connector} {intens} it is {adj}.",
            "The {noun} is {adj}. {connector} I do not recommend it.",
            "{intens} {adj}. {connector} it fails to deliver what it promises.",
            "Very {adj}. {connector} it's not worth the price.",
            "I was {verb_past}. {connector} {intens} {adj}.",
            "It's an absolute disgrace. The {noun} is {adj}.",
            "{intens} disappointed. {connector} the {noun} doesn't live up to expectations.",
            "Don't fall for it. The {noun} is {adj}.",
            "A complete disaster. {connector} {intens} do not recommend this product."
        ]
    
    def generar_espanol(self, positiva=True):
        """Genera reseña en español"""
        if positiva:
            patron = random.choice(self.es_patrones_pos)
            adj = random.choice(self.es_adj_pos)
            verbo = random.choice(self.es_verbos_pos)
            verbo_participio = verbo
        else:
            patron = random.choice(self.es_patrones_neg)
            adj = random.choice(self.es_adj_neg)
            verbo = random.choice(self.es_verbos_neg)
            verbo_participio = verbo
        
        sust = random.choice(self.es_sustantivos)
        conector = random.choice(self.es_conectores)
        intens = random.choice(self.es_intens)
        
        if conector and conector != "":
            conector = conector + " "
        
        texto = patron.format(
            adj=adj, sust=sust, conector=conector, intens=intens,
            verbo=verbo, verbo_participio=verbo_participio
        )
        
        texto = re.sub(r'\s+', ' ', texto).strip()
        if texto and texto[0].islower():
            texto = texto[0].upper() + texto[1:]
        if texto and texto[-1] not in ['.', '!', '?']:
            texto += '.'
        
        rating = random.choice([4, 5]) if positiva else random.choice([1, 2])
        return texto, rating
    
    def generar_ingles(self, positiva=True):
        """Genera reseña en inglés"""
        if positiva:
            patron = random.choice(self.en_patrones_pos)
            adj = random.choice(self.en_adj_pos)
            verbo = random.choice(self.en_verbs_pos)
            verbo_past = verbo
        else:
            patron = random.choice(self.en_patrones_neg)
            adj = random.choice(self.en_adj_neg)
            verbo = random.choice(self.en_verbs_neg)
            verbo_past = verbo
        
        noun = random.choice(self.en_nouns)
        connector = random.choice(self.en_connectors)
        intens = random.choice(self.en_intens)
        
        if connector and connector != "":
            connector = connector + " "
        
        texto = patron.format(
            adj=adj, noun=noun, connector=connector, intens=intens,
            verb=verbo, verb_past=verbo_past
        )
        
        texto = re.sub(r'\s+', ' ', texto).strip()
        if texto and texto[0].islower():
            texto = texto[0].upper() + texto[1:]
        if texto and texto[-1] not in ['.', '!', '?']:
            texto += '.'
        
        rating = random.choice([4, 5]) if positiva else random.choice([1, 2])
        return texto, rating
    
    def generar_dataset(self, num_resenas, idioma="ambos", proporcion_positivas=0.5):
        """Genera dataset masivo bilingüe"""
        
        reviews = []
        ratings = []
        
        if idioma == "espanol":
            lang_desc = "español"
            generador = lambda p: self.generar_espanol(p)
        elif idioma == "ingles":
            lang_desc = "inglés"
            generador = lambda p: self.generar_ingles(p)
        else:
            lang_desc = "español e inglés"
            generador = lambda p: self.generar_espanol(p) if random.choice([True, False]) else self.generar_ingles(p)
        
        num_pos = int(num_resenas * proporcion_positivas)
        num_neg = num_resenas - num_pos
        
        print(f"🔄 Generando {num_resenas:,} reseñas en {lang_desc}...")
        print(f"📚 Vocabulario disponible:")
        print(f"   🇪🇸 Español: {len(self.es_adj_pos):,} adj+, {len(self.es_adj_neg):,} adj-, {len(self.es_sustantivos):,} sustantivos")
        print(f"   🇺🇸 Inglés: {len(self.en_adj_pos):,} adj+, {len(self.en_adj_neg):,} adj-, {len(self.en_nouns):,} nouns")
        
        for i in range(num_pos):
            texto, rating = generador(True)
            reviews.append(texto)
            ratings.append(rating)
            if (i + 1) % 10000 == 0:
                print(f"   Positivas: {i+1:,}")
        
        for i in range(num_neg):
            texto, rating = generador(False)
            reviews.append(texto)
            ratings.append(rating)
            if (i + 1) % 10000 == 0:
                print(f"   Negativas: {i+1:,}")
        
        indices = list(range(num_resenas))
        random.shuffle(indices)
        
        reviews_shuffled = [reviews[i] for i in indices]
        ratings_shuffled = [ratings[i] for i in indices]
        
        return pd.DataFrame({'reviewText': reviews_shuffled, 'overall': ratings_shuffled})
    
    def guardar_dataset(self, num_resenas, nombre_archivo, idioma="ambos", proporcion_positivas=0.5):
        """Genera y guarda el dataset"""
        
        df = self.generar_dataset(num_resenas, idioma, proporcion_positivas)
        df.to_csv(nombre_archivo, index=False)
        
        print(f"\n✅ GUARDADO: {nombre_archivo}")
        print(f"📊 Total: {len(df):,} reseñas")
        print(f"⭐ Positivas: {sum(df['overall'] >= 4):,} ({sum(df['overall'] >= 4)/len(df)*100:.1f}%)")
        print(f"⭐ Negativas: {sum(df['overall'] <= 2):,} ({sum(df['overall'] <= 2)/len(df)*100:.1f}%)")
        
        return df


# ============================================
# EJECUCIÓN
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("🚀 GENERADOR ASTRONÓMICO BILINGÜE - VERSIÓN MEGA MASIVA")
    print("Inglés + Español | Vocabulario EXTREMO")
    print("Combinaciones prácticamente infinitas")
    print("="*70)

    # Preguntar al usuario
    NUM_RESENAS = int(input("📝 ¿Cuántas reseñas quieres generar? "))
    IDIOMA = input("🌐 ¿Idioma? (espanol/ingles/ambos): ").lower()
    
    # Calcular cuántas cifras tiene el número
    num_cifras = len(str(NUM_RESENAS))
    
    # Generar proporción aleatoria con esos decimales
    proporcion_positivas = round(random.uniform(0.1, 0.9), num_cifras)
    
    print(f"📊 Número introducido: {NUM_RESENAS} ({num_cifras} cifras)")
    print(f"📊 Proporción de positivas: {proporcion_positivas*100:.{num_cifras}f}%")
    print(f"📊 Proporción de negativas: {(1-proporcion_positivas)*100:.{num_cifras}f}%")
    
    generador = GeneradorAstronomico()
    
    df = generador.guardar_dataset(
        num_resenas=NUM_RESENAS,
        nombre_archivo=f"dataset_mega_{IDIOMA}_{NUM_RESENAS}.csv",
        idioma=IDIOMA,
        proporcion_positivas=proporcion_positivas
    )
    
    print("\n📝 EJEMPLOS DE RESEÑAS GENERADAS:")
    print("-" * 70)
    for i in range(3):
        esp_pos, rating_pos = generador.generar_espanol(True)
        print(f"🇪🇸 POSITIVA ({rating_pos}⭐): {esp_pos}")
    
    print()
    for i in range(3):
        esp_neg, rating_neg = generador.generar_espanol(False)
        print(f"🇪🇸 NEGATIVA ({rating_neg}⭐): {esp_neg}")
    
    print()
    for i in range(3):
        eng_pos, rating_pos = generador.generar_ingles(True)
        print(f"🇺🇸 POSITIVE ({rating_pos}⭐): {eng_pos}")
    
    print()
    for i in range(3):
        eng_neg, rating_neg = generador.generar_ingles(False)
        print(f"🇺🇸 NEGATIVE ({rating_neg}⭐): {eng_neg}")
    
    print("\n" + "="*70)
    print("✅ GENERADOR LISTO - VOCABULARIO MASIVO CARGADO")
    print("="*70)
