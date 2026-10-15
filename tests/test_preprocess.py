"""Tests for src/preprocess.py: at least one test per rule, plus edge cases."""

import os
import sys

# Make `src/` importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import (  # noqa: E402
    clean_text,
    collapse_repeated_mentions,
    lowercase,
    remove_hashtag_symbols,
    remove_leading_rt,
    remove_lone_underscores,
    remove_variation_selector,
    replace_mentions,
    replace_urls,
    shorten_repeated_letters,
    space_out_emoji,
    tidy_spaces,
)


# ---------------------------------------------------------------- Rule 1: variation selector
def test_rule1_variation_selector_removed():
    assert remove_variation_selector("I ❤️ you") == "I ❤ you"


def test_rule1_heart_with_and_without_selector_become_identical():
    assert clean_text("❤️") == clean_text("❤") == "❤"


# ---------------------------------------------------------------- Rule 2: leading RT
def test_rule2_leading_rt_removed():
    assert remove_leading_rt("RT @ a hello") == "@ a hello"
    assert clean_text("RT @ Rahul hello") == "@user hello"


def test_rule2_rt_is_case_sensitive():
    assert remove_leading_rt("rt hello") == "rt hello"      # lowercase "rt" is a normal word
    assert remove_leading_rt("Rt hello") == "Rt hello"


def test_rule2_rt_must_be_a_whole_token():
    assert remove_leading_rt("RTX is good") == "RTX is good"
    assert remove_leading_rt("RT: hello") == "RT: hello"


def test_rule2_rt_in_middle_of_sentence_kept():
    assert clean_text("please RT this") == "please rt this"
    assert clean_text("please rt this") == "please rt this"


def test_rule2_repeated_leading_rt_removed():
    assert clean_text("RT RT hello") == "hello"


# ---------------------------------------------------------------- Rule 3: URLs
def test_rule3_tco_dot_shape():
    assert replace_urls("see https // t . co / AbC123xyz now") == "see http now"


def test_rule3_tco_space_shape():
    assert replace_urls("see https // t co / AbC123xyz now") == "see http now"


def test_rule3_tco_joined_shape():
    assert replace_urls("see https // tco / AbC123xyz now") == "see http now"


def test_rule3_plain_http_scheme():
    assert replace_urls("see http // t . co / AbC123xyz") == "see http"


def test_rule3_other_domain_link():
    assert replace_urls("see https // www . youtube . com / watch ? v = abc123 now") == "see http now"


def test_rule3_cut_off_urls():
    for cut in ["https …", "https // t …", "https // t . c …", "https // t . co /…", "https //…",
                "https // t co / abc …", "https // tco …"]:
        assert replace_urls("good luck " + cut) == "good luck http", cut


def test_rule3_url_followed_by_text_keeps_text():
    assert replace_urls("https // t . co / AbC123xyz via @ YouTube") == "http via @ YouTube"


def test_rule3_two_urls_in_a_row():
    assert replace_urls("https // tco / aaa https // t . co / bbb") == "http http"


def test_rule3_plain_punctuation_after_http_not_eaten():
    assert replace_urls("see https . ok") == "see http . ok"


# ---------------------------------------------------------------- Rule 4: mentions
def test_rule4_split_mention():
    assert replace_mentions("hi @ nehantics bye") == "hi @user bye"


def test_rule4_mention_split_at_underscore():
    assert replace_mentions("@ Nehr _ who hi") == "@user hi"
    assert replace_mentions("@ a _ b _ c hi") == "@user hi"


def test_rule4_name_starting_with_underscore():
    assert replace_mentions("@_ YogendraYadav hi") == "@user hi"


def test_rule4_joined_mention():
    assert replace_mentions("hi @nehantics bye") == "hi @user bye"


def test_rule4_name_with_digits():
    assert replace_mentions("@ 454dkhan hi") == "@user hi"


def test_rule4_email_like_text_untouched():
    assert replace_mentions("mail a@b.com now") == "mail a@b.com now"


# ---------------------------------------------------------------- Rule 5: repeated mentions
def test_rule5_repeated_mentions_collapsed():
    assert collapse_repeated_mentions("@user @user @user hi") == "@user hi"
    assert clean_text("@ a @ b @ c hi") == "@user hi"


def test_rule5_mentions_apart_not_collapsed():
    assert clean_text("@ a hello @ b") == "@user hello @user"


def test_rule5_mentions_separated_by_trailing_underscore_collapse():
    assert clean_text("@ a _ @ b hi") == "@user hi"


# ---------------------------------------------------------------- Rule 6: hashtags and underscores
def test_rule6_hash_token_removed_word_kept():
    # the helper leaves a space where the "#" was; rule 10 (tidy_spaces) removes it
    assert tidy_spaces(remove_hashtag_symbols("# HappyBirthday to you")) == "HappyBirthday to you"


def test_rule6_standalone_underscores_removed():
    assert tidy_spaces(remove_lone_underscores("a _ b __ c")) == "a b c"


def test_rule6_glued_hashtag_and_c_sharp_untouched():
    assert remove_hashtag_symbols("#चम्पारण and c# #####") == \
        "#चम्पारण and c# #####"


def test_rule6_hash_after_emoji_and_underscore_hash():
    assert clean_text("\U0001F449# Word") == "\U0001F449 word"
    assert clean_text("x #_ thoda _ sa") == "x thoda sa"


# ---------------------------------------------------------------- Rule 7: lowercase
def test_rule7_lowercase():
    assert lowercase("Bahut ACCHA Hai") == "bahut accha hai"


# ---------------------------------------------------------------- Rule 8: repeated letters
def test_rule8_repeated_letters_collapsed_to_two():
    assert shorten_repeated_letters("bahuttt sooooo") == "bahutt soo"


def test_rule8_two_repeats_untouched():
    assert shorten_repeated_letters("hello baap") == "hello baap"


def test_rule8_digits_not_collapsed():
    assert shorten_repeated_letters("1000 and 2000000") == "1000 and 2000000"
    assert clean_text("pay 1000 rupees") == "pay 1000 rupees"


def test_rule8_punctuation_not_collapsed():
    assert shorten_repeated_letters("wow !!!! ....") == "wow !!!! ...."


# ---------------------------------------------------------------- Rule 9: emoji spacing
def test_rule9_glued_emoji_separated():
    assert space_out_emoji("\U0001F614\U0001F614") == " \U0001F614  \U0001F614 "
    assert clean_text("sad\U0001F614\U0001F614!") == "sad \U0001F614 \U0001F614 !"


def test_rule9_skin_tone_emoji_stays_whole():
    thumbs = "\U0001F44D\U0001F3FD"          # 👍🏽
    assert clean_text("ok " + thumbs) == "ok " + thumbs
    assert clean_text(thumbs + thumbs) == thumbs + " " + thumbs


def test_rule9_joined_family_emoji_stays_whole():
    family = "\U0001F468‍\U0001F469‍\U0001F467"   # 👨‍👩‍👧
    assert clean_text("love" + family) == "love " + family


def test_rule9_flags_stay_whole():
    flags = "\U0001F1EE\U0001F1F3\U0001F1F5\U0001F1F0"       # 🇮🇳🇵🇰
    assert clean_text(flags) == "\U0001F1EE\U0001F1F3 \U0001F1F5\U0001F1F0"


def test_rule9_text_without_emoji_unchanged():
    assert space_out_emoji("plain text") == "plain text"


# ---------------------------------------------------------------- Rule 10: spaces
def test_rule10_spaces_collapsed_and_stripped():
    assert tidy_spaces("  a   b \t c  ") == "a b c"


# ---------------------------------------------------------------- Edge cases and things we must NOT do
def test_digits_not_collapsed():
    assert clean_text("2000000") == "2000000"


def test_devanagari_text_unchanged():
    text = "मेरे देश की मिट्टी"
    assert clean_text(text) == text


def test_empty_string():
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_already_clean_text_unchanged():
    text = "yaar ye movie bahut acchi hai \U0001F60D @user http"
    assert clean_text(text) == text


def test_punctuation_accents_and_spellings_kept():
    assert clean_text("café nhi hai !!! ... ?") == "café nhi hai !!! ... ?"


def test_no_nfkc_normalisation():
    fancy = "\U0001D4F5\U0001D4F8\U0001D4FF\U0001D4EE"       # fancy-font "love"
    assert clean_text(fancy) == fancy


def test_full_example_all_rules_together():
    raw = ("RT @ Nehr _ who @ Other # HappyBirthday sooo goood ❤️\U0001F614\U0001F614 "
           "https // t . co / AbC123xyz")
    assert clean_text(raw) == "@user happybirthday soo good ❤ \U0001F614 \U0001F614 http"


def test_clean_text_is_idempotent():
    samples = [
        "RT @ Nehr _ who # Hit https // t . co / AbC123xyz \U0001F614\U0001F614",
        "@ a _ @ b  @ c hello",
        "good luck https // t . co /…",
        "x https // t . co / abc . Gud",
        "\U0001F449# Word \U0001F338@ Kathrin _ brid",
        "..#_ thoda _ sa",
        "@#@# mai",
        "RT RT rt",
        "",
    ]
    for text in samples:
        once = clean_text(text)
        assert clean_text(once) == once, text


def test_rule3_url_glued_to_a_word():
    assert clean_text("Yadavhttps // youtu . be / hTpid2R6zQE Ye hai") == "yadav http ye hai"


def test_rule4_cut_off_mention_at_end_of_tweet():
    assert clean_text("hi @ aajtak @…") == "hi @user"


def test_rule4_censored_swear_words_are_not_mentions():
    assert clean_text("gaa @#@# mai @$ har") == "gaa @#@# mai @$ har"
