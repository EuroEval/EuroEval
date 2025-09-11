# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==3.5.0",
#     "huggingface-hub==0.34.4",
#     "pandas==2.2.0",
#     "tqdm==4.66.3",
# ]
# ///

"""Create the legal completeness detection dataset."""

import hashlib
import random
import re
import typing as t

import pandas as pd
from datasets import Dataset, DatasetDict, Split
from huggingface_hub import HfApi
from tqdm.auto import tqdm

NUM_SAMPLES = 512
COMPLETE_CONTRACT_PROBABILITY = 0.3
MAX_REQUIRED_ELEMENTS_TO_REMOVE = 3
NICE_TO_HAVE_ELEMENT_REMOVAL_PROBABILITY = 0.3
SECTION_IDX_TAG = "[SECTION-IDX]"


COMPLETE_CONTRACT_EXAMPLE = f"""
## Ansættelseskontrakt

Mellem undertegnede

<company-name>## Greenwise Tech ApS</company-name>

<location>Vesterbrogade 42, 3. sal 1620 København V CVR-nr.: 40321088 (herefter kaldet virksomheden)</location>

og medundertegnede

## Maria Andersen

Tårnbyvej 18 2770 Kastrup (herefter kaldet medarbejderen)

indgås følgende:

<start-date>
## {SECTION_IDX_TAG}. Tiltrædelse

Medarbejderen tiltræder stillingen med virkning fra 10.11.2022
</start-date>

## {SECTION_IDX_TAG}. Arbejdssted - og arbejdsopgaver

<location>Arbejdsstedet er Vesterbrogade 42, 3. sal</location>

Medarbejderen er ansat <title>som administrativ medarbejder</title> <company-name>hos Greenwise Solutions ApS</company-name> og skal primært varetage følgende opgaver:

## 1. Daglig administration

- Håndtering af indgående og udgående post, e-mails og telefonopkald.
- Kalenderstyring og mødebooking for virksomhedens ledelse og medarbejdere.
- Udarbejdelse og arkivering af dokumenter, herunder breve, referater og præsentationer.

## 2. Økonomisk administration

- Registrering og håndtering af fakturaer, bilag og udgifter i virksomhedens regnskabssystem.
- Assistere med månedlige afstemninger og forberedelse af materiale til revisor.
- Opfølgning på betalinger og udsendelse af rykkere ved manglende betaling.

## 3. HR- og personalerelaterede opgaver

- Vedligeholdelse af medarbejderdata og opdatering af personalehåndbog.
- Assistere med onboarding af nye medarbejdere og koordinering af medarbejderaktiviteter.
- Håndtering af syge- og feriemeldinger.

## 4. Praktiske og ad hoc-opgaver

- Indkøb af kontorartikler og forplejning.
- Koordinering af interne arrangementer og firmaevents.
- Andre opgaver efter aftale med ledelsen.

Medarbejderens arbejdsopgaver kan løbende blive justeret i takt med virksomhedens behov.

<trial-period>
Ansættelsesforholdet er omfattet af en prøvetid på 3 måneder fra tiltrædelsestidspunktet. I prøvetiden kan ansættelsesforholdet opsiges af begge parter med 14 dages varsel til en hvilken som helst dag.
</trial-period>

<salary>
## {SECTION_IDX_TAG}. Løn

Den årlige løn vil være kr. 420.000 brutto (før skat, pension og lovpligtige bidrag), som udbetales månedsvis bagud med kr. 35.000 brutto. Beløbet indsættes på medarbejderens anviste konto den sidste hverdag i måneden.

Lønnen tages op til forhandling en gang årligt 1. oktober.
</salary>

<pension>
## {SECTION_IDX_TAG}. Pension

Der etableres en pensionsordning i PFA Pension. Til pensionsordningen indbetaler virksomheden 8% og medarbejderen 4% i tillæg til alle faste løndele som pensionsbidrag.
</pension>

<social-security>
## {SECTION_IDX_TAG}. Sociale sikringsbidrag

Virksomheden indbetaler lovpligtige sociale sikringsbidrag i henhold til gældende dansk lovgivning, herunder ATP, barselsfonden samt andre relevante bidrag.
</social-security>

<work-time>
## {SECTION_IDX_TAG}. Arbejdstid

Den ugentlige arbejdstid udgør 37 timer, eksklusive frokostpause.

Medarbejderen tilrettelægger selv sin arbejdstid, der placeres indenfor virksomhedens normale arbejdstid.

Der er ved fastsættelse af lønnen i pkt. 4 taget højde for honorering af merarbejde, hvorfor medarbejderen ikke vil modtage yderligere honorering. Ved pålagt overarbejde eller overarbejde af et særligt omfang indgås aftale med ledelsen om afspadsering eller betaling.
</work-time>

<holiday>
## {SECTION_IDX_TAG}. Ferie

Der tilkommer medarbejderen ferie med løn i overensstemmelse med ferielovens regler.

Det særlige ferietillæg i henhold til ferieloven udbetales med 2.48%.

Juleaftensdag og Nytårsaftensdag er fridage med fuld løn.
</holiday>

<extra-holiday>
## {SECTION_IDX_TAG}. Feriefridage

Der tilkommer medarbejderen yderligere 5 feriefridage pr. ferieår. Feriefridagene følger ferielovens bestemmelser.

Feriefridage der ikke er afholdt ved ferieårets udløb kan på medarbejderens anmodning udbetales eller overføres til næste ferieår.

Ikke afholdt feriefridage udbetales kontant i forbindelse med fratræden.
</extra-holiday>

<training>
## {SECTION_IDX_TAG}. Efteruddannelse

Medarbejderen har ret til relevant efter- og videreuddannelse i overensstemmelse med virksomhedens behov og gældende regler i overenskomsten. Efteruddannelse aftales løbende mellem virksomheden og medarbejderen.
</training>

<illness>
## {SECTION_IDX_TAG}. Sygdom

Hvis medarbejderen bliver fraværende fra sit arbejde, skal meddelelse herom så hurtigt som muligt gives til arbejdsgiveren.

Ved fravær ud over 3 dage kan arbejdsgiveren forlange, at medarbejderen ved friattest (lægeerklæring) skal dokumentere, at udeblivelsen skyldes sygdom.

Hvis arbejdsgiveren forlanger dokumentation for sygefraværet, betales denne af arbejdsgiveren.

Medarbejderen har krav på sin fulde sædvanlige løn under sygdom.
</illness>

<parental-leave>
## {SECTION_IDX_TAG}. Graviditet, barsel og adoption

Medarbejderen er berettiget til orlov i forbindelse med graviditet, fødsel og adoption i overensstemmelse med barsellovens regler herom.

Virksomheden betaler fuld løn til mor i følgende periode:

- 4 uger før forventet fødsel
- 24 uger efter fødsel

Virksomheden betaler fuld løn til far/medmor i:

- 24 uger efter fødsel

Faren/medmoren har ret til at placere 2 af ugerne i sammenhæng inden for de første 10 uger efter fødsel. De øvrige uger kan placeres efter den 10. uge efter fødslen.

Ovennævnte finder desuden fuldt ud anvendelse i tilfælde af adoption. Fuld løn i 4 uger før forventet modtagelse ydes, hvis det er et krav fra adoptionsmyndighederne, at adoptivbarnet skal hentes i udlandet.
</parental-leave>

## {SECTION_IDX_TAG}. Barns sygdom

Medarbejderen har ret til 2 dages frihed med løn ved barns sygdom.

## {SECTION_IDX_TAG}. Omsorgsdage

Medarbejdere med børn har ret til 2 børneomsorgsdage med løn pr. barn. Dagene tildeles ved påbegyndelsen af nyt kalenderår til og med det kalenderår, hvori barnet fylder 7 år.

Retten til omsorgsdage gælder fra ansættelsestidspunktet. Er omsorgsdagene ikke afholdt ved årets udgang bortfalder disse, ligesom medarbejderen ikke kompenseres for ikke afholdte omsorgsdage i forbindelse med fratræden.

## {SECTION_IDX_TAG}. Multimedier

Virksomheden stiller mobiltelefon, internetopkobling og pc til rådighed for medarbejderen. Multimedier kan benyttes både arbejdsmæssigt og privat. Virksomheden afholder omkostningerne ved etableringen samt de løbende udgifter.

Medarbejderen er opmærksom på, at denne bliver beskattet efter gældende regler af disse goder.

## {SECTION_IDX_TAG}. Rejser og repræsentation

Medarbejderens udgifter til transport, rejser, repræsentation, deltagelse i kurser m.v., der sker i virksomhedens interesse, dækkes af virksomheden efter regning. Medarbejderen har dog ret til at få udbetalt et passende forskud til afholdelsen af disse udgifter.

## {SECTION_IDX_TAG}. Tavshedspligt og loyalitetsforpligtigelse

Medarbejderen har tavshedspligt såvel under ansættelsen som efter fratræden med hensyn til forhold, som må betegnes som forretningshemmeligheder.

Medarbejderen er således ikke berettiget til at viderebringe oplysninger til tredjemand, som ikke er alment kendt i branchen. Der henvises til lov om forretningshemmeligheder.

<termination>
## {SECTION_IDX_TAG}. Opsigelse

Ansættelsesforhold kan fra begge parters side opsiges med det i funktionærlovens fastsatte varsel.

Opsigelsesvarslet fra virksomhedens side er:

- indtil udgangen af 5 måneders ansættelse udgør opsigelsesvarslet 1 måned
- indtil 2 år og 9 måneders ansættelse udgør opsigelsesvarslet 3 måneder
- indtil 5 år og 8 måneders ansættelse udgør opsigelsesvarslet 4 måneder
- indtil 8 år og 7 måneders ansættelse udgør opsigelsesvarslet 5 måneder
- og herefter med 6 måneders varsel

Fra medarbejderens side kan opsigelse ske med 1 måneds varsel til ophør med en måneds udgang.

Opsigelsen sker til udløbet af en kalendermåned, og skal ske skriftligt. Opsigelsen skal være modtageren i hænde senest den sidste dag i måneden.
</termination>

<collective-agreement>
## {SECTION_IDX_TAG}. Kollektiv overenskomst

Ansættelsesforholdet er omfattet af HK-overenskomsten for administrative medarbejdere i den private sektor, indgået mellem Dansk Erhverv og HK Privat.
</collective-agreement>

## {SECTION_IDX_TAG}. Retsgrundlag

Ansættelsesforholdet er reguleret af dansk ret. I det omfang ovenstående ikke stiller medarbejderen bedre, finder funktionær- og ferielovens bestemmelser anvendelse for ansættelsesforholdet.

København, den 01.11.2022

Medarbejderen _____________________________________

Virksomheden _____________________________________
""".strip()  # noqa: E501


REQUIRED_ELEMENTS = {
    "company-name": "Manglende angivelse af virksomhedens navn",
    "location": "Manglende angivelse af arbejdsstedets beliggenhed eller hvor "
    "arbejdet udføres",
    "title": "Manglende beskrivelse af medarbejderes stilling, titel, rang eller "
    "jobkategori",
    "start-date": "Manglende angivelse af ansættelsesforholdets begyndelsestidspunkt",
    "termination": "Manglende vilkår for opsigelse med varsler fra både ansatte og "
    "virksomhed",
    "salary": "Manglende angivelse af ansattes løn",
    "work-time": "Manglende angivelse af den ansattes arbejdstid",
    "social-security": "Manglende angivelse af hvilke bidrag til social sikring, som "
    "er knyttet til ansættelsesforholdet",
}

NICE_TO_HAVE_ELEMENTS = {
    "trial-period": "Manglende angivelse af vilkår og varihed for en prøvetid for "
    "ansættelse",
    "holiday": "Manglende definition af retten til ferie",
    "extra-holiday": "Manglende definition af retten til feriefridage",
    "illness": "Manglende beskrivelse af rettigheder i forbindelse med sygdom",
    "parental-leave": "Manglende beskrivelse af rettigheder i forbindelse med "
    "graviditet og barsel",
    "collective-agreement": "Manglende angivelse af hvilken overneskomst og aftale "
    "som ansættelsesforholdet er omfattet af",
    "pension": "Manglende definition af pensionsvilkår",
    "training": "Manglende angivelse af retten til efteruddannelse",
}


def main() -> None:
    """Create the legal completeness detection dataset and upload it to the HF Hub."""
    samples: list[dict[str, t.Any]] = list()
    hashes: set[str] = set()
    with tqdm(total=NUM_SAMPLES, desc="Generating samples") as pbar:
        while len(samples) < NUM_SAMPLES:
            contract, missing_required_elements, missing_nice_to_have_elements = (
                generate_contract()
            )
            contract_hash = hashlib.md5(contract.encode("utf-8")).hexdigest()
            if contract_hash in hashes:
                continue

            hashes.add(contract_hash)

            if len(missing_required_elements) == 0:
                label = "Kontrakten er fuldstændig."
            else:
                label = "Manglende lovpligtige elementer:\n" + "\n".join(
                    f"{idx}. {element}"
                    for idx, element in enumerate(missing_required_elements, start=1)
                )
            if len(missing_nice_to_have_elements) > 0:
                label += "\n\nManglende ikke-lovpligtige elementer:\n" + "\n".join(
                    f"{idx}. {element}"
                    for idx, element in enumerate(
                        missing_nice_to_have_elements, start=10
                    )
                )

            sample = dict(
                text=contract,
                label=label,
                missing_required_elements=missing_required_elements,
                missing_nice_to_have_elements=missing_nice_to_have_elements,
                num_missing_required_elements=len(missing_required_elements),
                num_missing_nice_to_have_elements=len(missing_nice_to_have_elements),
                is_complete=len(missing_required_elements) == 0,
            )
            samples.append(sample)
            pbar.update(1)

    # Create the dataset from the records
    df = pd.DataFrame.from_records(samples)
    split = Dataset.from_pandas(df, split=Split.TEST)
    dataset = DatasetDict(test=split)

    # Upload dataset
    dataset_id = "EuroEval/legal-completeness-detection"
    HfApi().delete_repo(dataset_id, repo_type="dataset", missing_ok=True)
    dataset.push_to_hub(dataset_id, private=True)


def generate_contract() -> tuple[str, list[str], list[str]]:
    """Generate a contract.

    Returns:
        A tuple (contract, missing_required_elements, missing_nice_to_have_elements),
        where `contract` is the generated contract, and the other two lists contain the
        missing required and nice-to-have elements, respectively.
    """
    contract = COMPLETE_CONTRACT_EXAMPLE
    missing_required_elements = []
    missing_nice_to_have_elements = []

    # Decide if the contract should be complete or not
    is_complete = random.random() < COMPLETE_CONTRACT_PROBABILITY

    # If not complete, remove some required elements
    if not is_complete:
        num_elements_to_remove = random.randint(1, MAX_REQUIRED_ELEMENTS_TO_REMOVE)
        elements_to_remove = random.sample(
            list(REQUIRED_ELEMENTS.keys()), num_elements_to_remove
        )
        for element in elements_to_remove:
            contract = re.sub(
                rf"<{element}>.*?</{element}>", "", contract, flags=re.DOTALL
            )
            missing_required_elements.append(REQUIRED_ELEMENTS[element])

    # Randomly remove some nice-to-have elements
    for element, description in NICE_TO_HAVE_ELEMENTS.items():
        if random.random() < NICE_TO_HAVE_ELEMENT_REMOVAL_PROBABILITY:
            contract = re.sub(
                rf"<{element}>.*?</{element}>", "", contract, flags=re.DOTALL
            )
            missing_nice_to_have_elements.append(description)

    # Clean up the contract, which removes the remaining XML tags, sets up the section
    # numbering, and general whitespace cleanup
    contract = clean_up_contract(contract=contract)

    return contract, missing_required_elements, missing_nice_to_have_elements


def clean_up_contract(contract: str) -> str:
    """Clean up the contract by removing extra newlines and spaces.

    Args:
        contract:
            The contract to clean up.

    Returns:
        The cleaned up contract.
    """
    # Remove all XML tags that might be left
    contract = re.sub(pattern=r"</?[^>]+>", repl="", string=contract, flags=re.DOTALL)

    # Remove multiple newlines
    contract = re.sub(pattern=r"(\s*\n\s*){2,}", repl="\n\n", string=contract)

    # Remove multiple spaces
    contract = re.sub(pattern=r" {2,}", repl=" ", string=contract)

    # Strip leading and trailing whitespace
    contract = contract.strip()

    # Set up section numbering
    section_idx = 1
    while SECTION_IDX_TAG in contract:
        contract = contract.replace(SECTION_IDX_TAG, str(section_idx), 1)
        section_idx += 1

    return contract


if __name__ == "__main__":
    main()
