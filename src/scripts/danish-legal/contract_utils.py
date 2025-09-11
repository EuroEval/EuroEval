# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "pandas==2.2.0",
#     "openai==1.107.1",
#     "click==8.2.1",
#     "pydantic==2.11.7",
#     "python-dotenv==1.1.1",
#     "tqdm==4.67.1",
#     "pyarrow==21.0.0",
# ]
# ///

"""Utility functions related to Danish contracts."""

import datetime as dt
import logging
import os
import random
import re
import typing as t
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from openai import LengthFinishReasonError, OpenAI
from pydantic import BaseModel, Field, ValidationError
from tqdm.auto import tqdm

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("contract_utils")

load_dotenv()

ALL_POSTAL_CODES: dict[int, str] = (
    pd.read_csv(Path(__file__).parent / "postal_codes.tsv")
    .set_index("postal_code")["city"]
    .to_dict()
)
POSTAL_CODE_PATTERN: re.Pattern[str] = re.compile(
    pattern=re.compile(r"|".join([str(code) for code in ALL_POSTAL_CODES.keys()]))
)

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


@click.command()
@click.option(
    "--num-contracts",
    "-n",
    type=int,
    default=10,
    help="The number of contracts to generate.",
)
@click.option(
    "--openai-model-id",
    "-m",
    type=str,
    default="gpt-4.1",
    help="The OpenAI model ID to use for generation.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=3,
    help="The number of contract infos to generate in each API call.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default=Path("generated_contracts"),
    help="The directory to store the generated contracts.",
)
def main(
    num_contracts: int, batch_size: int, openai_model_id: str, output_dir: Path
) -> None:
    """Generate and store Danish employment contracts."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = 0

    contract_infos: list[EmploymentContractInfo] = list()
    with tqdm(total=num_contracts, desc="Generating contract infos") as pbar:
        while len(contract_infos) < num_contracts:
            try:
                new_infos = generate_contract_infos(
                    num_contracts=min(num_contracts - len(contract_infos), batch_size),
                    openai_model_id=openai_model_id,
                )
            except LengthFinishReasonError:
                batch_size -= 1
                logger.warning(
                    "Response too long from model, retrying with smaller batch size..."
                )
                continue
            contract_infos.extend(new_infos)
            pbar.update(min(len(new_infos), pbar.total - pbar.n))

    for i, contract_info in enumerate(
        tqdm(contract_infos, desc="Generating contracts")
    ):
        contract = generate_employment_contract(
            contract_info=contract_info, openai_model_id=openai_model_id
        )
        output_path = output_dir / f"employment_contract_{i + 1 + offset:03d}.md"
        while output_path.exists():
            offset += 1
            output_path = output_dir / f"employment_contract_{i + 1 + offset:03d}.md"
        with output_path.open("w", encoding="utf-8") as f:
            f.write(contract)


class Address(BaseModel):
    """A postal address in Denmark."""

    road_name: t.Annotated[str, Field(description="The name of the road/street.")]
    road_number: t.Annotated[int, Field(ge=1, description="The road/street number.")]
    floor: (
        t.Annotated[
            int,
            Field(ge=0, le=10, description="The floor number (0 for ground floor)."),
        ]
        | None
    )
    side: (
        t.Annotated[
            str, Field(pattern=r"(tv|th|mf)\.", description="The side (tv., th., mf.).")
        ]
        | None
    )
    postal_code: t.Annotated[int, Field(description="The 4-digit postal code.")]
    city: t.Annotated[
        str, Field(description="The city corresponding to the postal code.")
    ]

    def model_post_init(self, _: None) -> None:
        """Post-initialisation validation."""
        self.postal_code = random.choice(list(ALL_POSTAL_CODES.keys()))
        self.city = ALL_POSTAL_CODES[int(self.postal_code)]
        if self.floor is None:
            self.side = None

    def __repr__(self) -> str:
        """Return the address as a formatted string."""
        address_str = f"{self.road_name} {self.road_number}"
        if self.floor is not None:
            address_str += f", {self.floor if self.floor > 0 else 'st'}."
        if self.side is not None:
            address_str += f" {self.side}"
        address_str += f", {self.postal_code} {self.city}"
        return address_str


class EmploymentContractInfo(BaseModel):
    """Information to be contained in a Danish employment contract."""

    company_name: t.Annotated[
        str, Field(description="The legal name of the employing company.")
    ]
    company_cvr_number: t.Annotated[
        int,
        Field(
            ge=10_000_000, le=99_999_999, description="The CVR number of the company."
        ),
    ]
    company_address: t.Annotated[
        Address, Field(description="The registered address of the employing company.")
    ]
    office_address: t.Annotated[
        Address,
        Field(
            description="The address of the office where the employee will work. "
            "Can be the same as the company address."
        ),
    ]

    employee_name: t.Annotated[str, Field(description="The full name of the employee.")]
    employee_address: t.Annotated[
        Address, Field(description="The residential address of the employee.")
    ]
    employee_title: t.Annotated[
        str, Field(description="The job title of the employee.")
    ]
    employee_task_description: t.Annotated[
        str, Field(description="A description of the tasks of the employee.")
    ]

    start_date: t.Annotated[
        dt.date, Field(description="The date the employment starts.")
    ]
    annual_salary_dkk: t.Annotated[
        int, Field(ge=10_000, le=2_000_000, description="The annual salary in DKK.")
    ]
    monthly_salary_dkk: t.Annotated[
        int, Field(ge=1, le=200_000, description="The monthly salary in DKK.")
    ]
    trial_period_months: (
        t.Annotated[
            int,
            Field(ge=1, le=6, description="The length of the trial period in months."),
        ]
        | None
    )
    termination_notice_days: t.Annotated[
        int, Field(ge=1, le=100, description="The termination notice period in days.")
    ]
    weekly_work_hours: t.Annotated[
        int, Field(ge=1, le=80, description="The number of work hours per week.")
    ]
    extra_holiday_week: t.Annotated[
        bool,
        Field(description="Whether the employee is entitled to an extra holiday week."),
    ]

    maternal_leave_weeks_before_birth: t.Annotated[
        int, Field(ge=0, le=10, description="Weeks of maternal leave before birth.")
    ]
    maternal_leave_weeks_after_birth: t.Annotated[
        int, Field(ge=0, le=52, description="Weeks of maternal leave after birth.")
    ]
    paternal_leave_weeks_after_birth: t.Annotated[
        int, Field(ge=0, le=52, description="Weeks of paternal leave after birth.")
    ]
    child_illness_days: t.Annotated[
        int, Field(ge=0, le=10, description="Days off for child illness per year.")
    ]

    pension_company: t.Annotated[
        str, Field(description="The name of the pension company.")
    ]
    pension_employer_contribution: t.Annotated[
        float,
        Field(ge=0.0, le=1.0, description="The employer's pension contribution rate."),
    ]
    pension_employee_contribution: t.Annotated[
        float,
        Field(ge=0.0, le=1.0, description="The employee's pension contribution rate."),
    ]

    confidentiality_clause: t.Annotated[
        bool,
        Field(description="Whether there is a confidentiality clause in the contract."),
    ]
    collective_agreement_company: t.Annotated[
        str | None,
        Field(
            description="The name of the collective agreement the company is part of, "
            "if any."
        ),
    ]

    contract_date: t.Annotated[
        dt.date, Field(description="The date the contract was created.")
    ]

    def model_post_init(self, _: None) -> None:
        """Post-initialisation validation."""
        self.contract_date = self.start_date - dt.timedelta(days=random.randint(7, 90))
        self.monthly_salary_dkk = self.annual_salary_dkk // 12


class EmploymentContractInfos(BaseModel):
    """A list of EmploymentContractInfo objects."""

    contract_infos: t.Annotated[
        list[EmploymentContractInfo],
        Field(description="A list of employment contract information objects."),
    ]


def generate_contract_infos(
    num_contracts: int, openai_model_id: str
) -> list[EmploymentContractInfo]:
    """Generate random but realistic information for a Danish employment contract.

    Args:
        num_contracts:
            The number of contract infos to generate.
        openai_model_id:
            The OpenAI model ID to use for generation.

    Returns:
        The generated employment contract information.

    Raises:
        RuntimeError:
            If the response from the model is too long.
    """
    client = OpenAI()

    contract_info_fields = "\n".join(
        [
            f"- {field_name}: {field_info.description}"
            for field_name, field_info in EmploymentContractInfo.model_fields.items()
        ]
    )

    system_prompt = f"""
        Du er en ekspert i at generere realistiske oplysninger til ansættelseskontrakter
        i Danmark. Du skriver kun på dansk. Følgende felter skal inkluderes:

        {contract_info_fields}
    """.replace(" {2,}", "").strip()
    user_prompt = f"""
        Generér {num_contracts} sæt realistiske og sammenhængende oplysninger til danske
        ansættelseskontrakter. Du må gerne gøre dem forskellige ift. brancher og vilkår.

        Du skal returnere en JSON dictionary med nøglen `contract_infos`, der indeholder
        en liste med oplysninger til de {num_contracts} kontrakter, hvor hvert element
        er en JSON dictionary med alle felterne ovenfor.
    """.replace(" {2,}", "").strip()

    while True:
        try:
            response = client.beta.chat.completions.parse(
                messages=[
                    dict(role="system", content=system_prompt),
                    dict(role="user", content=user_prompt),
                ],
                model=openai_model_id,
                temperature=1.0,
                max_completion_tokens=32_768,
                response_format=EmploymentContractInfos,
            )
            contract_infos_obj = response.choices[0].message.parsed
            if contract_infos_obj is None:
                logger.error(
                    "Failed to parse contract info from model response. Retrying..."
                )
            else:
                return contract_infos_obj.contract_infos
        except ValidationError as e:
            logger.info(
                f"Validation error when parsing contract info:\n{e.json(indent=2)}\n"
                "Retrying..."
            )
            continue


def generate_employment_contract(
    contract_info: EmploymentContractInfo, openai_model_id: str
) -> str:
    """Generate a Danish employment contract based on the provided information.

    Args:
        contract_info:
            The information to be included in the contract.
        openai_model_id:
            The OpenAI model ID to use for generation.

    Returns:
        The generated employment contract as a string.

    Raises:
        ValueError:
            If no contract is generated by the model.
    """
    xml_tags_in_example = set(re.findall(r"</?([^>]+)>", COMPLETE_CONTRACT_EXAMPLE))
    xml_tags_str = "\n".join(["- " + tag for tag in xml_tags_in_example])
    system_prompt = """
        Du er en ekspert i at skrive ansættelseskontrakter i Danmark. Du skriver kun
        på dansk.
    """.replace(" {2,}", "").strip()
    user_prompt = f"""
        Skriv en ansættelseskontrakt i Markdown baseret på følgende oplysninger.

        <kontrakt-oplysninger>
        {contract_info}
        </kontrakt-oplysninger>

        Her er et eksempel på en komplet ansættelseskontrakt, som du skal bruge som
        reference. Den indeholder alle de nødvendige sektioner og oplysninger, der
        typisk findes i en dansk ansættelseskontrakt.

        <kontrakt-eksempel>
        {COMPLETE_CONTRACT_EXAMPLE}
        </kontrakt-eksempel>

        Følg strukturen og formatet i det vedhæftede eksempel på en komplet
        ansættelseskontrakt, men du må gerne ændre indholdet samt rækkefølgen af afsnit,
        hvis det giver mening, så det ikke er en kopi af eksemplet.

        Din kontrakt skal indeholde følgende XML tags, som skal udfyldes med de
        relevante oplysninger, ligesom i eksemplet:

        {xml_tags_str}

        Alle XML-tags skal have indhold, de kan ikke bare være tomme. Du må ikke
        tilføje andre XML-tags i kontrakten.

        Teksten skal give mening, selvom et XML-tag (med indhold) fjernes. Så fx er
        dette ikke tilladt, da hvis vi fjerner <title> tagget, så giver sætningen ikke
        mening:

        <bad-use-of-xml>
        <title>Stillingen som Softwareudvikler</title> hos Nordic Tech Solutions ApS
        indebærer følgende hovedopgaver:
        </bad-use-of-xml>

        Derudover ser vi også virksomhedsnavnet, men der mangler et <company-name> tag
        omkring det, så det er heller ikke tilladt. Her er den korrekte brug:

        <good-use-of-xml>
        Stillingen <title>som Softwareudvikler</title> <company-name>hos Nordic Tech
        Solutions ApS</company-name> indebærer følgende hovedopgaver:
        </good-use-of-xml>

        Hvis vi fjerner <company-name> tagget og/eller <title> tagget, så giver
        sætningen stadig mening.

        Du også blive ved med at bruge {SECTION_IDX_TAG} til at markere afsnit, der skal
        nummereres senere.

        Du skal kun skrive kontrakten uden yderligere forklaringer.
    """.replace(" {2,}", "").strip()

    client = OpenAI()

    while True:
        response = client.chat.completions.create(
            messages=[
                dict(role="system", content=system_prompt),
                dict(role="user", content=user_prompt),
            ],
            model=openai_model_id,
            temperature=1.0,
            max_completion_tokens=4096,
        )
        contract = response.choices[0].message.content
        if contract is None:
            raise ValueError("No contract generated by the model.")
        contract = contract.strip()

        # Remove ``` backticks if present
        contract = re.sub(r"^```(markdown)?\n+", "", contract)
        contract = re.sub(r"\n+```$", "", contract)

        # Remove XML comments
        contract = re.sub(r"<!--.*?-->", "", contract, flags=re.DOTALL)

        # Replace snake case XML tags with kebab-case
        xml_tags_in_contract = set(re.findall(r"</?([^>]+)>", contract))
        snake_case_tags = {tag for tag in xml_tags_in_contract if "_" in tag}
        for tag in snake_case_tags:
            kebab_case_tag = tag.replace("_", "-")
            contract = re.sub(
                pattern=rf"</?{tag}>",
                repl=lambda m: m.group(0).replace(tag, kebab_case_tag),
                string=contract,
            )

        # Extract XML tags before and after
        xml_tags_in_contract = set(re.findall(r"</?([^>]+)>", contract))
        xml_tags_in_example = set(re.findall(r"</?([^>]+)>", COMPLETE_CONTRACT_EXAMPLE))
        empty_xml_tags = {
            tag
            for tag in xml_tags_in_contract
            if re.search(rf"<{tag}>\s*</{tag}>", contract, flags=re.DOTALL)
        }

        # Check if we need to retry generation
        hallucinated_tags = xml_tags_in_contract - xml_tags_in_example
        missing_tags = xml_tags_in_example - xml_tags_in_contract
        have_to_retry = bool(hallucinated_tags or missing_tags or empty_xml_tags)
        if hallucinated_tags:
            logger.warning(f"Hallucinated XML tags in contract: {hallucinated_tags}.")
        if missing_tags:
            logger.warning(f"Missing XML tags in contract: {missing_tags}.")
        if empty_xml_tags:
            logger.warning(f"Empty XML tags in contract: {empty_xml_tags}.")
        if have_to_retry:
            logger.info("Retrying generation of contract...")
            continue
        break

    return contract


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = os.environ["MUCH_OPENAI_API_KEY"]
    main()
