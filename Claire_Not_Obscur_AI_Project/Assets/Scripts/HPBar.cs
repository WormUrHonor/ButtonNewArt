using JetBrains.Annotations;
using NUnit.Framework;
using System.Collections.Generic;
using TMPro;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;


public class HPBar : MonoBehaviour
{
    public Health health;
    public TMP_Text playerName;

    public GameObject hpIcons;
    public Sprite HpIconSprite;

    [SerializeField]
    private List<Image> images;

    [Header("Sound Effects")]
    public AudioClip hitSound;
    public AudioClip hurtSound;
    public AudioClip deathSound;
    public AudioClip deathGong;
    public AudioClip breathe;

    private AudioSource audioSource;

    private void Start()
    {
        audioSource = gameObject.AddComponent<AudioSource>();
        playerName.text = health.player == PlayerID.Player1 ? "Claire" : "Cliare";

        images = new List<Image>();
        Image[] childImages = hpIcons.GetComponentsInChildren<Image>();
        for (int i = 0; i < childImages.Length; i++)
        {
            Vector3 translation = health.player == PlayerID.Player1 ? new Vector3(0,0,0): new Vector3(-3.25f * 135,0,0);
            childImages[i].rectTransform.Translate(translation);
            childImages[i].sprite = HpIconSprite;
            images.Add(childImages[i]);
        }
    }

    public void DecrementHP()
    {

        for (int i = health.maxHP - health.currentHP; i > 0; i--)
        {
            if (health.player == PlayerID.Player1)
            {
                images[images.Count - i].color = new Color(1f, 1f, 1f, 0.3f);
            }
            else
            {
                images[i - 1].color = new Color(1f, 1f, 1f, 0.3f);
            }
        }

        if (hurtSound != null && health.currentHP > 0)
        {
            audioSource.PlayOneShot(hitSound);
            StartCoroutine(PlayHurtSoundDelayed(0.15f));
        }

        if (breathe != null && health.currentHP == 1)
        {
            if (audioSource.clip != breathe)
            {
                audioSource.clip = breathe;
                audioSource.loop = true;
                audioSource.volume = 0.4f;
                audioSource.Play();
                audioSource.volume = 1f;   // restore for SFX
            }
        }
        else if (audioSource.clip == breathe && health.currentHP != 1)
        {

            audioSource.Stop();
            audioSource.loop = false;
            audioSource.clip = null;
        }

        if (deathSound != null && health.currentHP <= 0)
        {
            audioSource.PlayOneShot(deathSound, 0.7f);
        }

        if (deathGong != null && health.currentHP <= 0)
        {
            audioSource.PlayOneShot(deathGong);
        }

        if (health.currentHP <= 0)
        {
            playerName.text = "DEAD";
        }

    }
    private System.Collections.IEnumerator PlayHurtSoundDelayed(float delay)
    {
        yield return new WaitForSeconds(delay);
        audioSource.PlayOneShot(hurtSound);
    }


    public void IncrementHP()
    {

        for (int i = 0; i < health.currentHP; i++)
        {
            if (health.player == PlayerID.Player1)
            {
                images[i].color = new Color(1f, 1f, 1f, 1f);
            }
            else
            {
                images[images.Count - 1 - i].color = new Color(1f, 1f, 1f, 1f);
            }
        }

    }
}
