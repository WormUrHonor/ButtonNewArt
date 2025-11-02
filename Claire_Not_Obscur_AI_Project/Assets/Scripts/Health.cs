using System.Collections.Generic;
using TMPro;
using Unity.Mathematics;
//using UnityEditor.SearchService;
using UnityEngine;
using UnityEngine.SceneManagement;

public enum PlayerID
{
    Player1, Player2
}


public class Health : MonoBehaviour
{
    public PlayerID player;
    [SerializeField] public HPBar hpBar;
    public int maxHP;

    [SerializeField]
    public int currentHP;

    public bool hasWon;
    public bool hasLost;

    public GameObject WinScreen;
    public GameObject LoseScreen;

    private void Start()
    {
        currentHP = maxHP;
        Time.timeScale = 1;
        WinScreen.SetActive(false);
        LoseScreen.SetActive(false);
    }

    public int DecrementHP()
    {
        
        currentHP--;
        currentHP = math.clamp(currentHP, 0, maxHP);
        hpBar.DecrementHP();
        return currentHP;
    }

    public int IncrementHP()
    {
        currentHP++;
        currentHP = math.clamp(currentHP, 0, maxHP);
        hpBar.IncrementHP();
        return currentHP;
    }

    void Update()
    {
        if(currentHP <=0 && SceneManager.sceneCount < 2)
        {
            if(player == PlayerID.Player1)
            {
                hasLost = true;
            }

            else
            {
                hasWon = true;
            }

            Time.timeScale = 0;

            WinScreen.SetActive(hasWon);
            LoseScreen.SetActive(hasLost);

        }
    }
}
